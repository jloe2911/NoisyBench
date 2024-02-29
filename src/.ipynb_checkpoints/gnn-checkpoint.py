import pandas as pd
import numpy as np
import operator
import random

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear, to_hetero
from torch_geometric.utils import negative_sampling

class GCN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_dim)
        self.conv2 = GCNConv(-1, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_dim)
        self.conv2 = GATConv((-1, -1), output_dim, add_self_loops=False)
        self.lin2 = Linear(-1, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x
    
class GNN():
    def __init__(self):
        self.model = None
        self.epochs = 300
        self.node_embed_size = 200
        self.hidden_dim = 200
        self.output_dim = 200
        self.seed = 10
        torch.manual_seed(self.seed)
    
    def _train(self, GNN_variant, data, nodes, threshold):        
        pos_edge_index = data.train_pos_edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_neg_samples = int(pos_edge_index.size()[1] * threshold))
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        num_nodes = len(nodes)
        
        self.node_embeds = torch.rand(num_nodes, self.node_embed_size)
        
        if GNN_variant == 'GCN':
            self.model = GCN(self.hidden_dim, self.output_dim)
        elif GNN_variant == 'GraphSAGE':
            self.model = GraphSAGE(self.hidden_dim, self.output_dim)
        elif GNN_variant == 'GAT':
            self.model = GAT(self.hidden_dim, self.output_dim)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)    
        targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])])
        #targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.ones(neg_edge_index.shape[1])]) 
        #in case we want to add negative triples as part of the training triples,
        #we must label them as positive triples to add noise to the dataset
        
        for i in range(self.epochs+1):
            self.model.train()
            optimizer.zero_grad()
            
            embeds = self.model(self.node_embeds, edge_index)
                
            u = torch.index_select(embeds, 0, edge_index[0, :])
            v = torch.index_select(embeds, 0, edge_index[1, :])
            pred = torch.sum(u * v, dim=-1)
            pred = (pred - pred.min()) / (pred.max() - pred.min())
            
            loss = mse_loss(pred, targets)
            loss.backward()
            optimizer.step()
            
            if i % self.epochs == 0:
                print(f'Epoch: {i}, Loss: {loss:.4f}')
                
    def _eval(self, GNN_variant, data, threshold):
        with torch.no_grad():
            self.model.eval()

            pos_edge_index = data.test_pos_edge_index
            neg_edge_index = negative_sampling(pos_edge_index, num_neg_samples = int(pos_edge_index.size()[1] * threshold))
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

            output = self.model(self.node_embeds, edge_index)
            
            hits1, hits10 = eval_hits(data=data,
                                      tail_pred=1,
                                      output=output,
                                      max_num=100)
            
            print(f'hits@1: {hits1:.3f}, hits@10: {hits10:.3f}')
            
###HELPER FUNCIONS### 

def mse_loss(pred, target):
    return (pred - target.to(pred.dtype)).pow(2).mean()

def eval_hits(data, tail_pred, output, max_num):    
    top1 = 0
    top10 = 0
    n = data.test_pos_edge_index.size(1)

    for idx in range(n):
        if tail_pred == 1:
            x = torch.index_select(output, 0, data.test_pos_edge_index[0, idx])
        else:
            x = torch.index_select(output, 0, data.test_pos_edge_index[1, idx])
        
        candidates, candidates_embeds = sample_negative_edges_idx(idx=idx,
                                                                  data=data,
                                                                  tail_pred=tail_pred,
                                                                  output=output,
                                                                  max_num=max_num)

        distances = torch.cdist(candidates_embeds, x, p=2)
        dist_dict = {cand: dist for cand, dist in zip(candidates, distances)} 

        sorted_dict = dict(sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=True))
        sorted_keys = list(sorted_dict.keys())

        ranks_dict = {sorted_keys[i]: i for i in range(0, len(sorted_keys))}
        rank = ranks_dict[data.test_pos_edge_index[1, idx].item()]
        
        if rank <= 1:
            top1 += 1
        if rank <= 10:
            top10 += 1
    return top1/n, top10/n

def sample_negative_edges_idx(idx, data, tail_pred, output, max_num):
    num_neg_samples = 0
    candidates = []
    nodes = list(range(data.test_pos_edge_index.size(1)))
    random.shuffle(nodes)

    while num_neg_samples < max_num:    
        if tail_pred == 1:
            t = nodes[num_neg_samples]
            h = data.test_pos_edge_index[0, idx].item()
            if h not in data.test_pos_edge_index[0] or t not in data.test_pos_edge_index[1]:
                candidates.append(t)
        else: 
            t = data.test_pos_edge_index[1, idx].item()
            h = nodes[num_neg_samples]
            if h not in data.test_pos_edge_index[0] or t not in data.test_pos_edge_index[1]:
                candidates.append(h)
        num_neg_samples += 1
    candidates_embeds = torch.index_select(output, 0, torch.tensor(candidates))

    if tail_pred == 1:
        true_tail = data.test_pos_edge_index[1, idx]
        candidates.append(true_tail.item())
        candidates_embeds = torch.concat([candidates_embeds, torch.index_select(output, 0, true_tail)])
    else:
        true_head = data.test_pos_edge_index[0, idx]
        candidates.append(true_head.item())
        candidates_embeds = torch.concat([candidates_embeds, torch.index_select(output, 0, true_head)])
    return candidates, candidates_embeds