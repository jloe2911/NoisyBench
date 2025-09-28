import os 
import operator
import random
import math
from collections import defaultdict
import rdflib
from rdflib import RDF, RDFS

import torch
import torch.nn.functional as F
from torch.nn import Parameter

import torch_geometric
from torch_geometric.nn import GAE, RGCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.utils import get_data, get_namespace, save_results

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
    
class GNN():
    def __init__(self, device, num_nodes, num_relations):
        self.device = device
        self.hidden_channels = 200
        self.model = GAE(RGCNEncoder(num_nodes, self.hidden_channels, num_relations),
                         DistMultDecoder(num_relations, self.hidden_channels)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.seed = 10
        torch.manual_seed(self.seed)

    def _train(self, data):        
        self.model.train()
        self.optimizer.zero_grad()
        
        z = self.model.encode(data.train_pos_edge_index, data.train_edge_type)

        pos_out = self.model.decode(z, data.train_pos_edge_index, data.train_edge_type)

        neg_edge_index = negative_sampling(data.train_pos_edge_index, num_neg_samples = data.train_pos_edge_index.size(1))
        neg_out = self.model.decode(z, neg_edge_index, data.train_edge_type)

        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()

        return float(loss)

    def _eval(self, data, target_type, multiple):
        with torch.no_grad():
            self.model.eval()

            new_data = get_specific_test_edge_type(data, target_type, multiple)

            output = self.model.encode(new_data.test_pos_edge_index, new_data.test_edge_type)

            mrr, hits_at_1, hits_at_5, hits_at_10 = eval_hits(edge_index=new_data.test_pos_edge_index,
                                                              tail_pred=1,
                                                              output=output,
                                                              max_num=100,
                                                              device=self.device)
                        
            return mrr, hits_at_1, hits_at_5, hits_at_10
            
###HELPER FUNCIONS### 

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

def eval_hits(edge_index, tail_pred, output, max_num, device):    
    mrr = 0
    mean_rank = 0
    rank_vals = []
    top1 = 0
    top5 = 0
    top10 = 0
    if edge_index.size(1) <= 100: n = edge_index.size(1)
    else: n = 100

    for idx in range(n):
        if tail_pred == 1:
            x = torch.index_select(output, 0, edge_index[0, idx])
        else:
            x = torch.index_select(output, 0, edge_index[1, idx])
        
        candidates, candidates_embeds = sample_negative_edges_idx(idx=idx,
                                                                  edge_index=edge_index,
                                                                  tail_pred=tail_pred,
                                                                  output=output,
                                                                  max_num=max_num,
                                                                  device=device)

        distances = torch.cdist(candidates_embeds, x, p=2)
        dist_dict = {cand: dist for cand, dist in zip(candidates, distances)} 

        sorted_dict = dict(sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=True))
        sorted_keys = list(sorted_dict.keys())

        ranks_dict = {sorted_keys[i]: i for i in range(0, len(sorted_keys))}
        if tail_pred == 1:
            rank = ranks_dict[edge_index[1, idx].item()]
        else:
            rank = ranks_dict[edge_index[0, idx].item()]
        
        mrr += 1/(rank+1)
        mean_rank += rank
        rank_vals.append(rank)
        if rank <= 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
        if rank <= 10:
            top10 += 1
    return mrr/n, top1/n, top5/n, top10/n 

def sample_negative_edges_idx(idx, edge_index, tail_pred, output, max_num, device):
    num_neg_samples = 0
    candidates = []
    nodes = list(range(edge_index.max()))
    random.shuffle(nodes)

    while num_neg_samples < max_num:    
        if tail_pred == 1:
            t = nodes[num_neg_samples]
            h = edge_index[0, idx].item()
            if h not in edge_index[0] or t not in edge_index[1]:
                candidates.append(t)
        else: 
            t = edge_index[1, idx].item()
            h = nodes[num_neg_samples]
            if h not in edge_index[0] or t not in edge_index[1]:
                candidates.append(h)
        num_neg_samples += 1
    candidates_embeds = torch.index_select(output, 0, torch.tensor(candidates).to(device))

    if tail_pred == 1:
        true_tail = edge_index[1, idx]
        candidates.append(true_tail.item())
        candidates_embeds = torch.concat([candidates_embeds, torch.index_select(output, 0, true_tail)])
    else:
        true_head = edge_index[0, idx]
        candidates.append(true_head.item())
        candidates_embeds = torch.concat([candidates_embeds, torch.index_select(output, 0, true_head)])
    return candidates, candidates_embeds.to(device)

def rdf_to_edge_index(g: rdflib.Graph, node2id: dict, rel2id: dict):
    """Convert RDF triples to edge_index and edge_type tensors"""
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

def get_specific_test_edge_type(data, target_type, multiple):
    data_copy = data.clone()   
    if multiple:
        mask = torch.isin(
        data_copy['test_edge_type'], 
        torch.tensor(target_type, dtype=data_copy['test_edge_type'].dtype)
    )
    else:   
        mask = data_copy['test_edge_type'] == target_type
    data_copy.test_pos_edge_index = data_copy['test_pos_edge_index'][:, mask]
    data_copy.test_edge_type = data_copy['test_edge_type'][mask]
    return data_copy

def train_gnn_reasoner(dataset_name, device, iteration=0, epochs=300):    
    g_train = rdflib.Graph()
    g_train.parse(f"datasets/{dataset_name}_train.owl")

    g_test = rdflib.Graph()
    g_test.parse(f"datasets/{dataset_name}_test.owl")
    _, _, _, _, relations_dict_test = get_data(g_test)

    g_val = rdflib.Graph()
    g_val.parse(f"datasets/{dataset_name}_val.owl")

    # Initialize mapping dictionaries
    node2id = {}
    rel2id = {}

    # Convert RDF graphs to edge_index tensors
    train_edge_index, train_edge_type = rdf_to_edge_index(g_train, node2id, rel2id)
    val_edge_index, val_edge_type = rdf_to_edge_index(g_val, node2id, rel2id)
    test_edge_index, test_edge_type = rdf_to_edge_index(g_test, node2id, rel2id)

    # Create HeteroData object
    data = HeteroData(
        train_pos_edge_index=train_edge_index,
        train_edge_type=train_edge_type,
        val_pos_edge_index=val_edge_index,
        val_edge_type=val_edge_type,
        test_pos_edge_index=test_edge_index,
        test_edge_type=test_edge_type
    )

    nodes = list({n for g in [g_train, g_test, g_val] for n in list(g.subjects()) + list(g.objects())})
    relations = list({n for g in [g_train, g_test, g_val] for n in list(g.predicates())})

    model = GNN(device, len(nodes), len(relations))    
    for epoch in range(epochs+1):
        loss = model._train(data.to(device))
        if epoch % 50 == 0:
            logging.info(f'Epoch {epoch}, Loss: {loss:.4f}')

    os.makedirs("models/RGCN_reasoner", exist_ok=True)  
    torch.save(model, f'models/RGCN_reasoner/{dataset_name}_{iteration}')
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
        keys_list = [key for key in relations_dict_test.keys() if key.startswith(NS)]
        target_type = [relations_dict_test[key] for key in keys_list if key in relations_dict_test]
        multiple = True
    mrr, hits_at_1, hits_at_5, hits_at_10 = model._eval(data.to(device), target_type, multiple)
    return (mrr, hits_at_1, hits_at_5, hits_at_10)

def test_gnn_reasoner(dataset_name, model, data, relations_dict_test, device):
    logger.info("Testing ontology completion...")

    logger.info('Membership:')
    membership_metrics = compute_ranking_metrics(dataset_name, model, data, relations_dict_test, device, "test_membership") 
    mrr, hits_at_1, hits_at_5, hits_at_10 = membership_metrics
    logger.info(f'MRR: {mrr:.3f}, Hits@1: {hits_at_1:.3f}, Hits@5: {hits_at_5:.3f}, Hits@10: {hits_at_10:.3f}')      
    
    logger.info('Subsumption:')
    subsumption_metrics = compute_ranking_metrics(dataset_name, model, data, relations_dict_test, device, "test_subsumption")
    mrr, hits_at_1, hits_at_5, hits_at_10 = subsumption_metrics
    logger.info(f'MRR: {mrr:.3f}, Hits@1: {hits_at_1:.3f}, Hits@5: {hits_at_5:.3f}, Hits@10: {hits_at_10:.3f}')      
   
    logger.info('Link Prediction:')
    link_prediction_metrics = compute_ranking_metrics(dataset_name, model, data, relations_dict_test, device, "test_link_prediction")
    mrr, hits_at_1, hits_at_5, hits_at_10 = link_prediction_metrics
    logger.info(f'MRR: {mrr:.3f}, Hits@1: {hits_at_1:.3f}, Hits@5: {hits_at_5:.3f}, Hits@10: {hits_at_10:.3f}')      
    
    return subsumption_metrics, membership_metrics, link_prediction_metrics

def run_rgcn(device, experiments):
    os.makedirs(f'models/results/rgcn_reasoner/', exist_ok=True)
    for experiment in experiments:
        dataset_name = experiment['dataset_name']
        file_name = experiment['file_name']
        
        subsumption_results = []
        membership_results = []
        link_prediction_results = []
        
        for i in range(5):
            model, data, relations_dict_test = train_gnn_reasoner(dataset_name, device, iteration=i, epochs=300)
            
            logging.info(f'{file_name}:')
            metrics_subsumption, metrics_membership, metrics_link_prediction = test_gnn_reasoner(dataset_name, model, data, relations_dict_test, device)

            subsumption_results.append(metrics_subsumption)
            membership_results.append(metrics_membership)
            link_prediction_results.append(metrics_link_prediction)

        save_results(subsumption_results, membership_results, link_prediction_results, f'models/results/rgcn_reasoner/{file_name}.txt')

def run_rgcn_test(device, experiments):
    os.makedirs(f'models/results/rgcn_reasoner/', exist_ok=True)
    for experiment in experiments:
        dataset_name = experiment['dataset_name']
        file_name = experiment['file_name']
        
        subsumption_results = []
        membership_results = []
        link_prediction_results = []
        
        model, data, relations_dict_test = train_gnn_reasoner(dataset_name, device, iteration=0, epochs=300)
        
        logging.info(f'{file_name}:')
        metrics_subsumption, metrics_membership, metrics_link_prediction = test_gnn_reasoner(dataset_name, model, data, relations_dict_test, device)

        subsumption_results.append(metrics_subsumption)
        membership_results.append(metrics_membership)
        link_prediction_results.append(metrics_link_prediction)

        save_results(subsumption_results, membership_results, link_prediction_results, f'models/results/rgcn_reasoner/{file_name}_test.txt')

        break