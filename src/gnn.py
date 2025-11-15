import os
import math
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import negative_sampling, to_undirected
from torch_geometric.data import HeteroData

import rdflib
from rdflib import RDF, RDFS

from src.utils import get_data, get_namespace, save_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


############################
# Encoder & Decoder Models #
############################

class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_blocks=5)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_blocks=5)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


#####################
# GNN Core Wrapper  #
#####################

class GNN(torch.nn.Module):
    def __init__(self, seed, device, num_nodes, num_relations, rdf_type_id=None):
        super().__init__()
        self.device = device
        self.hidden_channels = 200
        self.encoder = RGCNEncoder(num_nodes, self.hidden_channels, num_relations)
        self.decoder = DistMultDecoder(num_relations, self.hidden_channels)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.rdf_type_id = rdf_type_id
        self.class_nodes = None  # filled dynamically
        torch.manual_seed(seed)

    def encode(self, data):
        return self.encoder(data.train_pos_edge_index, data.train_edge_type)

    def decode(self, z, edge_index, edge_type):
        return self.decoder(z, edge_index, edge_type)

    def _train(self, data):
        self.train()
        self.optimizer.zero_grad()
        z = self.encode(data)

        # --- collect class nodes for rdf:type ---
        if self.rdf_type_id is not None and self.class_nodes is None:
            mask = data.train_edge_type == self.rdf_type_id
            self.class_nodes = data.train_pos_edge_index[1, mask].unique()

        # --- positive samples ---
        pos_out = self.decode(z, data.train_pos_edge_index, data.train_edge_type)

        # --- relation-aware negative sampling ---
        neg_edge_index_list, neg_edge_type_list = [], []
        for rel in data.train_edge_type.unique():
            rel_mask = data.train_edge_type == rel
            pos_edges_rel = data.train_pos_edge_index[:, rel_mask]
            num_neg = pos_edges_rel.size(1)

            if self.rdf_type_id is not None and rel.item() == self.rdf_type_id:
                # RDF.type: only corrupt tail (class)
                src = pos_edges_rel[0]
                dst = self.class_nodes[torch.randint(0, len(self.class_nodes), (num_neg,), device=z.device)]
                neg_edges_rel = torch.stack([src, dst], dim=0)
            else:
                # Normal relations
                neg_edges_rel = negative_sampling(
                    pos_edges_rel, num_nodes=z.size(0), num_neg_samples=num_neg
                )

            neg_edge_index_list.append(neg_edges_rel)
            neg_edge_type_list.append(torch.full((num_neg,), rel, dtype=torch.long, device=z.device))

        neg_edge_index = torch.cat(neg_edge_index_list, dim=1)
        neg_edge_type = torch.cat(neg_edge_type_list, dim=0)
        neg_out = self.decode(z, neg_edge_index, neg_edge_type)

        # --- loss ---
        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        loss_ce = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + self.decoder.rel_emb.pow(2).mean()
        loss = loss_ce + 1e-2 * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return float(loss)
    
    def _eval(self, data, target_type, multiple):
        self.eval()
        with torch.no_grad():
            new_data = get_specific_test_edge_type(data, target_type, multiple)
            z = self.encoder(data.train_pos_edge_index, data.train_edge_type)
            mrr, hits1, hits5, hits10 = eval_hits_full_ranking(
                edge_index=new_data.test_pos_edge_index,
                edge_type=new_data.test_edge_type,
                z=z,
                rel_emb=self.decoder.rel_emb,
                rdf_type_id=self.rdf_type_id,
                class_nodes=self.class_nodes,
                device=self.device
            )
        return mrr, hits1, hits5, hits10


###########################################
#  Efficient Batched Evaluation Functions #
###########################################

@torch.no_grad()
def eval_hits_full_ranking(edge_index, edge_type, z, rel_emb,
                          rdf_type_id=None, class_nodes=None,
                          device='cpu'):
    num_edges = edge_index.size(1)
    num_nodes = z.size(0)
    if num_edges == 0:
        logging.warning("got 0 test edges — returning NaNs.")
        return float('nan'), float('nan'), float('nan'), float('nan')

    src, dst = edge_index
    rel = rel_emb[edge_type]
    pos_src, pos_dst = z[src], z[dst]
    pos_scores = torch.sum(pos_src * rel * pos_dst, dim=1)

    ranks = []
    for i in range(num_edges):
        r = edge_type[i].item()
        s = src[i].item()
        o = dst[i].item()

        # Get all candidate tail nodes for ranking
        if rdf_type_id is not None and r == rdf_type_id and class_nodes is not None:
            candidates = class_nodes
        else:
            candidates = torch.arange(num_nodes, device=device)

        # Filter out the true tail node from candidates
        candidates = candidates[candidates != o]

        # Embeddings
        z_s = z[s].unsqueeze(0)  # shape: [1, hidden_dim]
        rel_r = rel_emb[r].unsqueeze(0)  # shape: [1, hidden_dim]
        z_candidates = z[candidates]  # shape: [num_candidates, hidden_dim]

        # Compute scores for negatives
        neg_scores = torch.sum(z_s * rel_r * z_candidates, dim=1)  # [num_candidates]

        # Compare positive score with negatives
        rank = (neg_scores >= pos_scores[i]).sum().item() + 1  # rank starts at 1
        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float, device=device)
    mrr = (1.0 / ranks).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits5 = (ranks <= 5).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()

    return mrr, hits1, hits5, hits10


########################################
#   Dataset & Graph Processing Utils   #
########################################

def split_edges(data, test_ratio = 0.2, val_ratio = 0):    
    row, col = data.edge_index
    edge_type = data.edge_type

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))
    row, col, edge_type = row[perm], col[perm], edge_type[perm]

    r, c, e = row[:n_v], col[:n_v], edge_type[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_edge_type = e
    r, c, e = row[n_v:n_v + n_t], col[n_v:n_v + n_t], edge_type[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_edge_type = e
    r, c, e = row[n_v + n_t:], col[n_v + n_t:], edge_type[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_edge_type = e
    return data

def rdf_to_edge_index(g: rdflib.Graph, node2id: dict, rel2id: dict):
    src, dst, rel = [], [], []
    for s, p, o in g:
        if s not in node2id:
            node2id[s] = len(node2id)
        if o not in node2id:
            node2id[o] = len(node2id)
        if p not in rel2id:
            rel2id[p] = len(rel2id)
        src.append(node2id[s])
        dst.append(node2id[o])
        rel.append(rel2id[p])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)
    return edge_index, edge_type


def get_specific_test_edge_type(data, target_type, multiple, device='cpu'):
    """
    Filters the test edges for specific relation types.
    
    Args:
        data (HeteroData): The dataset containing test edges.
        target_type (int or list[int]): The target relation type(s) to evaluate.
        multiple (bool): Whether to treat target_type as multiple relations.
        device (str or torch.device): The device to place filtered tensors on.

    Returns:
        HeteroData: A copy of data with filtered test edges.
    """
    data_copy = data.clone()
    # Ensure target_type is a tensor
    if multiple:
        if not isinstance(target_type, (list, torch.Tensor)):
            target_type = [target_type]
        target_type_tensor = torch.tensor(target_type, dtype=data_copy['test_edge_type'].dtype, device=device)
        mask = torch.isin(data_copy['test_edge_type'], target_type_tensor)
    else:
        # Single relation type
        if isinstance(target_type, (list, tuple, torch.Tensor)):
            target_type = target_type[0]  # pick the first if wrapped in list
        mask = data_copy['test_edge_type'] == target_type

    # Debug: check how many edges remain
    num_edges = mask.sum().item()
    if num_edges == 0:
        logging.warning(f"get_specific_test_edge_type: no test edges found for target_type={target_type}")
    
    # Filter edges
    data_copy.test_pos_edge_index = data_copy['test_pos_edge_index'][:, mask].to(device)
    data_copy.test_edge_type = data_copy['test_edge_type'][mask].to(device)
    
    return data_copy


#############################################
#     Training, Evaluation, and Logging     #
#############################################

def train_gnn_reasoner(dataset_name, file_name, device, seed, iteration=0, epochs=500):
    g_train = rdflib.Graph()
    g_train.parse(f"datasets/{dataset_name}_train.owl")
    g_test = rdflib.Graph()
    g_test.parse(f"datasets/{file_name}_test.owl") # We add noise to the test set
    _, _, _, _, relations_dict_test = get_data(g_test)
    g_val = rdflib.Graph()
    g_val.parse(f"datasets/{dataset_name}_val.owl")

    node2id = {}
    rel2id = {}

    train_edge_index, train_edge_type = rdf_to_edge_index(g_train, node2id, rel2id)
    val_edge_index, val_edge_type = rdf_to_edge_index(g_val, node2id, rel2id)
    test_edge_index, test_edge_type = rdf_to_edge_index(g_test, node2id, rel2id)

    data = HeteroData()
    data.train_pos_edge_index = train_edge_index
    data.train_edge_type = train_edge_type
    data.val_pos_edge_index = val_edge_index
    data.val_edge_type = val_edge_type
    data.test_pos_edge_index = test_edge_index
    data.test_edge_type = test_edge_type

    model = GNN(seed, device, len(node2id), len(rel2id), rdf_type_id=rel2id.get(RDF.type, None)).to(device)

    for epoch in range(epochs + 1):
        loss = model._train(data.to(device))
        if epoch % 50 == 0:
            logging.info(f'Epoch {epoch}, Loss: {loss:.4f}')

    os.makedirs(f"models/RGCN_reasoner/{file_name}/", exist_ok=True)
    torch.save(model.state_dict(), f'models/RGCN_reasoner/{file_name}/run_{iteration}.pt')
    return model, data, relations_dict_test


def compute_ranking_metrics(dataset_name, model, data, relations_dict_test, device, mode):
    if 'subsumption' in mode:
        target_type = relations_dict_test[RDFS.subClassOf]
        multiple = False
    elif 'membership' in mode:
        target_type = relations_dict_test[RDF.type]
        multiple = False
    elif 'link_prediction' in mode:
        NS = get_namespace(dataset_name)
        keys_list = [key for key in relations_dict_test.keys() if str(key).startswith(NS)]
        target_type = [relations_dict_test[key] for key in keys_list if key in relations_dict_test]
        multiple = True
    return model._eval(data.to(device), target_type, multiple)


def test_gnn_reasoner(dataset_name, model, data, relations_dict_test, device):
    logger.info("Testing ontology completion...")

    logger.info('Membership:')
    membership_metrics = compute_ranking_metrics(dataset_name, model, data, relations_dict_test, device, "test_membership")
    logger.info(f'MRR: {membership_metrics[0]:.3f}, Hits@1: {membership_metrics[1]:.3f}, Hits@5: {membership_metrics[2]:.3f}, Hits@10: {membership_metrics[3]:.3f}')

    logger.info('Subsumption:')
    subsumption_metrics = compute_ranking_metrics(dataset_name, model, data, relations_dict_test, device, "test_subsumption")
    logger.info(f'MRR: {subsumption_metrics[0]:.3f}, Hits@1: {subsumption_metrics[1]:.3f}, Hits@5: {subsumption_metrics[2]:.3f}, Hits@10: {subsumption_metrics[3]:.3f}')

    logger.info('Link Prediction:')
    link_metrics = compute_ranking_metrics(dataset_name, model, data, relations_dict_test, device, "test_link_prediction")
    logger.info(f'MRR: {link_metrics[0]:.3f}, Hits@1: {link_metrics[1]:.3f}, Hits@5: {link_metrics[2]:.3f}, Hits@10: {link_metrics[3]:.3f}')

    return subsumption_metrics, membership_metrics, link_metrics


def run_rgcn(device, experiments):
    os.makedirs(f'models/results/rgcn_reasoner/', exist_ok=True)
    for experiment in experiments:
        dataset_name = experiment['dataset_name']
        file_name = experiment['file_name']

        subsumption_results = []
        membership_results = []
        link_prediction_results = []

        for i in range(5):
            seed = 42 + i
            model, data, relations_dict_test = train_gnn_reasoner(dataset_name, file_name, device, seed, iteration=i, epochs=300)
            logger.info(f'{file_name}:')
            metrics_subsumption, metrics_membership, metrics_link_prediction = test_gnn_reasoner(
                dataset_name, model, data, relations_dict_test, device
            )

            subsumption_results.append(metrics_subsumption)
            membership_results.append(metrics_membership)
            link_prediction_results.append(metrics_link_prediction)
        
        save_results(subsumption_results, membership_results, link_prediction_results,
                     f'models/results/rgcn_reasoner/{file_name}.txt')


def run_rgcn_test(device, experiments):
    os.makedirs(f'models/results/rgcn_reasoner/', exist_ok=True)
    for experiment in experiments:
        dataset_name = experiment['dataset_name']
        file_name = experiment['file_name']

        seed = 42 
        model, data, relations_dict_test = train_gnn_reasoner(dataset_name, file_name, device, seed, iteration=0, epochs=25)
        logger.info(f'{file_name}:')
        metrics_subsumption, metrics_membership, metrics_link_prediction = test_gnn_reasoner(
            dataset_name, model, data, relations_dict_test, device
        )

        save_results([metrics_subsumption], [metrics_membership], [metrics_link_prediction],
                     f'models/results/rgcn_reasoner/{file_name}_test.txt')
        break