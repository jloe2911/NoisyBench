import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict
import torch
from torch_geometric.data import HeteroData
import rdflib
from rdflib import URIRef, RDF
import mowl
mowl.init_jvm('10g')
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

def split_ontology(dataset_name, file_name, format_, train_ratio, add_noise):
    g = rdflib.Graph()
    g.parse(f'datasets/{dataset_name}.owl')  
    triples = list(g.triples((None, None, None))) 
    print(f'Triplets found in {dataset_name}.owl: %d' % len(g))

    g_subsumption = rdflib.Graph()
    g_subsumption.parse(f'datasets/test/{dataset_name}_classification.owl')
    subsumption_triples = list(g_subsumption.triples((None, URIRef('http://www.w3.org/2000/01/rdf-schema#subClassOf'), None))) 
    
    g_membership = rdflib.Graph()
    g_membership.parse(f'datasets/test/{dataset_name}_realisation.owl')
    membership_triples = list(g_membership.triples((None, RDF.type, None))) 

    if add_noise:   
        g_noise = rdflib.Graph()
        g_noise.parse(f'datasets/noise/{file_name}.owl', format=format_)  
        print(f'Triplets found in {file_name}.owl: %d' % len(g_noise))

    train_index = int(train_ratio * len(triples))
    train_triples = triples[:train_index]
    valid_triples = triples[train_index:]
    test_triples_subsumption = set(subsumption_triples) - set(train_triples) -set(valid_triples)
    test_triples_membership = set(membership_triples) - set(train_triples) -set(valid_triples)

    train_graph = rdflib.Graph()
    valid_graph = rdflib.Graph()
    test_graph = rdflib.Graph()
    test_subsumption_graph = rdflib.Graph()
    test_membership_graph = rdflib.Graph()

    for triple in train_triples:
        train_graph.add(triple)

    for triple in valid_triples:
        valid_graph.add(triple)

    # Tasks: subsumption, membership   
    for triple in test_triples_subsumption:
        test_graph.add(triple)
        test_subsumption_graph.add(triple)
            
    for triple in test_triples_membership:
        test_graph.add(triple)
        test_membership_graph.add(triple)

    # Add noisy triples to train_graph
    if add_noise:
        for triple in g_noise:
            train_graph.add(triple)

    print(f'Train Triplets found: %d' % len(train_graph))
    train_graph.serialize(destination=f"datasets/bin/{file_name}_train.owl")
    valid_graph.serialize(destination=f"datasets/bin/{file_name}_val.owl")
    test_graph.serialize(destination=f"datasets/bin/{file_name}_test.owl")
    print(f'Test Triplets (Membership) found: %d' % len(test_membership_graph))
    test_membership_graph.serialize(destination=f"datasets/bin/{file_name}_membership_test.owl")
    print(f'Test Triplets (Subsumption) found: %d' % len(test_subsumption_graph))
    test_subsumption_graph.serialize(destination=f"datasets/bin/{file_name}_subsumption_test.owl")

    return train_graph, valid_graph, test_graph, test_membership_graph, test_subsumption_graph

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

def get_experimets(dataset_name):
    
    if dataset_name == 'family': 
    
        experiments = [{'dataset_name' : 'family',
                        'file_name' : 'family',
                        'format_' : None,
                        'add_noise': False},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_noisy_gnn_100',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_noisy_gnn_1000',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_noisy_random_100',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_noisy_random_1000',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_noisy_disjoint_100',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_noisy_disjoint_1000',
                        'format_' : None,
                        'add_noise': True}]
 
    elif dataset_name == 'pizza':
    
        experiments = [{'dataset_name' : 'pizza',
                        'file_name' : 'pizza',
                        'format_' : None,
                        'add_noise': False},
                       {'dataset_name' : 'pizza',
                        'file_name' : 'pizza_noisy_random_100',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'pizza',
                        'file_name' : 'pizza_noisy_random_1000',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'pizza',
                        'file_name' : 'pizza_noisy_disjoint_100',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'pizza',
                        'file_name' : 'pizza_noisy_disjoint_1000',
                        'format_' : None,
                        'add_noise': True}]
        
    elif dataset_name == 'OWL2DL-1':
    
        experiments = [{'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1',
                        'format_' : None,
                        'add_noise': False},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_noisy_gnn_100',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_noisy_gnn_1000',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_noisy_random_100',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_noisy_random_1000',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_noisy_disjoint_100',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_noisy_disjoint_1000',
                        'format_' : None,
                        'add_noise': True}]
    
    return experiments