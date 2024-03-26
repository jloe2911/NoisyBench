import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict
import torch
from torch_geometric.data import HeteroData
import rdflib
from org.semanticweb.owlapi.model.parameters import Imports
from java.util import HashSet
from mowl.owlapi import OWLAPIAdapter

def get_data(g, nodes_dict, relations_dict):
    edge_data = defaultdict(list)
    for s, p, o in g.triples((None, None, None)):
        src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
        edge_data['edge_index'].append([src, dst])
        edge_data['edge_type'].append(rel)
    
    data = HeteroData(edge_index=torch.tensor(edge_data['edge_index'], dtype=torch.long).t().contiguous(),
                      edge_type=torch.tensor(edge_data['edge_type'], dtype=torch.long))
    
    return data

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

def copy_graph(g):
    new_g = rdflib.Graph()

    for triple in g:
        new_g.add(triple)
    
    return new_g

def split_ontology(file_name, format_, train_ratio, test_ratio):
    g = rdflib.Graph()
    g.parse(f'datasets/{file_name}.owl', format=format_)  
    print(f'Triplets found: %d' % len(g))

    triples = list(g.triples((None, None, None))) 
    random.shuffle(triples) 

    train_index = int(train_ratio * len(triples))
    test_index = int((train_ratio + test_ratio) * len(triples))

    train_triples = triples[:train_index]
    test_triples = triples[train_index:test_index]
    valid_triples = triples[test_index:]

    train_graph = rdflib.Graph()
    test_graph = rdflib.Graph()
    valid_graph = rdflib.Graph()

    for triple in train_triples:
        train_graph.add(triple)

    for triple in test_triples:
        test_graph.add(triple)
        
    for triple in valid_triples:
        valid_graph.add(triple)

    print(f'Train Triplets found: %d' % len(train_graph))
    train_graph.serialize(destination=f"datasets/bin/{file_name}_train.owl")
    print(f'Test Triplets found: %d' % len(test_graph))
    test_graph.serialize(destination=f"datasets/bin/{file_name}_test.owl")
    print(f'Valid Triplets found: %d' % len(valid_graph))
    valid_graph.serialize(destination=f"datasets/bin/{file_name}_val.owl")
    
    return train_graph, test_graph, valid_graph

def preprocess_ontology_el(ontology):
    tbox_axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))
    abox_axioms = ontology.getABoxAxioms(Imports.fromBoolean(True))
    new_tbox_axioms = HashSet() 
    for axiom in tbox_axioms:
        axiom_as_str = axiom.toString()
        if "ObjectHasValue" in axiom_as_str:
            continue
        elif "DataSomeValuesFrom" in axiom_as_str:
            continue
        elif "DataAllValuesFrom" in axiom_as_str:
            continue
        elif "DataHasValue" in axiom_as_str:
            continue
        elif "DataPropertyRange" in axiom_as_str:
            continue
        elif "DataPropertyDomain" in axiom_as_str:
            continue
        elif "FunctionalDataProperty" in axiom_as_str:
            continue
        elif "DisjointUnion" in axiom_as_str:
            continue
        elif "HasKey" in axiom_as_str:
            continue
        new_tbox_axioms.add(axiom)
    owl_manager = OWLAPIAdapter().owl_manager
    new_ontology = owl_manager.createOntology(new_tbox_axioms)
    new_ontology.addAxioms(abox_axioms)
    return new_ontology

def get_abox_data(dataset, dataset_type):
    if dataset_type == "train":
        ontology = dataset.ontology
    elif dataset_type == "valid":
        ontology = dataset.validation
    elif dataset_type == "test":
        ontology = dataset.testing
    abox = []
    for cls in dataset.classes:
        abox.extend(list(ontology.getClassAssertionAxioms(cls)))
    nb_individuals = len(dataset.individuals)
    nb_classes = len(dataset.classes)
    owl_indiv_to_id = dataset.individuals.to_index_dict()
    owl_class_to_id = dataset.classes.to_index_dict()
    labels = np.zeros((nb_individuals, nb_classes), dtype=np.int32)
    for axiom in abox:
        cls = axiom.getClassExpression()
        indiv = axiom.getIndividual()
        cls_id = owl_class_to_id[cls]
        indiv_id = owl_indiv_to_id[ind]
        labels[indiv_id, cls_id] = 1
    idxs = np.arange(nb_individuals)
    return torch.tensor(idxs), torch.FloatTensor(labels)