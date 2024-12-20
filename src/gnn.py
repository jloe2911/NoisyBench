import numpy as np
import operator
import random
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn import GAE, RGCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData

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

    def _eval(self, data):
        with torch.no_grad():
            self.model.eval()

            output = self.model.encode(data.test_pos_edge_index, data.test_edge_type)

            mrr, mean_rank, median_rank, hits5, hits10 = eval_hits(edge_index=data.test_pos_edge_index,
                                                                   tail_pred=1,
                                                                   output=output,
                                                                   max_num=100,
                                                                   device=self.device)
                        
            return mrr, mean_rank, median_rank, hits5, hits10
            
###HELPER FUNCIONS### 

def get_data(g):
    relations = list(set(g.predicates()))
    nodes = list(set(g.subjects()).union(set(g.objects())))
    relations_dict = {rel: i for i, rel in enumerate(relations)}
    nodes_dict = {node: i for i, node in enumerate(nodes)}

    edge_data = defaultdict(list)
    for s, p, o in g.triples((None, None, None)):
        src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
        edge_data['edge_index'].append([src, dst])
        edge_data['edge_type'].append(rel)
    
    data = HeteroData(edge_index=torch.tensor(edge_data['edge_index'], dtype=torch.long).t().contiguous(),
                      edge_type=torch.tensor(edge_data['edge_type'], dtype=torch.long))
    return data, nodes, nodes_dict, relations, relations_dict

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
    top5 = 0
    top10 = 0
    n = edge_index.size(1)

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
        if rank <= 5:
            top5 += 1
        if rank <= 10:
            top10 += 1
    return mrr/n, mean_rank/n,  np.median(rank_vals), top5/n, top10/n

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