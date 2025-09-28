import os
import torch
import rdflib
from rdflib import Namespace

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.gnn import get_data
from src.noise import train_gnn, get_disjoint_classes, get_disjoint_properties, add_triples_random, add_triples_gnn, add_triples_logical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # Override for consistency

def load_graphs(dataset_name: str):
    g = rdflib.Graph()
    g.parse(f'ontologies/{dataset_name}.owl', format='xml')

    g_modified = rdflib.Graph()
    g_modified.parse(f'ontologies/{dataset_name}_modified.owl', format='turtle') # we modified the TBOX by adding more disjoint classes and properties
    
    return g, g_modified

def add_noise_to_dataset(dataset_name: str, experiments: dict):
    g, g_modified = load_graphs(dataset_name)

    nodes, nodes_dict, relations, relations_dict = get_data(g)[1:]
    nodes_dict_rev = {v: k for k, v in nodes_dict.items()}
    relations_dict_rev = {v: k for k, v in relations_dict.items()}

    # Train GNN
    model, data = train_gnn(g, nodes, device, relations)
    os.makedirs("models", exist_ok=True)  
    torch.save(model, f'models/RGCN_{dataset_name}')

    # DL Noise
    all_disjoint_classes = get_disjoint_classes(g_modified)
    all_disjoint_properties = get_disjoint_properties(g_modified)

    os.makedirs("datasets/noise", exist_ok=True)

    # Generate noise files
    for noise_percentage in [0.25, 0.5, 0.75, 1.0]:
        # Random
        noisy_r, _ = add_triples_random(g, noise_percentage)
        noisy_r.serialize(f"datasets/noise/{dataset_name}_random_{noise_percentage}.owl", format='xml')
        logging.info(f"DONE - Random Noise - {noise_percentage}")

        # GNN
        model = torch.load(f'models/RGCN_{dataset_name}', weights_only=False)
        noisy_g, _ = add_triples_gnn(model, g, data, nodes_dict_rev, relations_dict_rev, device, noise_percentage)
        noisy_g.serialize(f"datasets/noise/{dataset_name}_gnn_{noise_percentage}.owl", format='xml')
        logging.info(f"DONE - Statistical Noise - {noise_percentage}")

        # Logical
        noisy_dl, _ = add_triples_logical(g_modified, noise_percentage, all_disjoint_classes, all_disjoint_properties)
        noisy_dl.serialize(f"datasets/noise/{dataset_name}_logical_{noise_percentage}.owl", format='xml')
        logging.info(f"DONE - Logical Noise - {noise_percentage}")

    # Create training datasets + noise
    for experiment in experiments[1:]: 
        
        dataset_name = experiment['dataset_name']
        file_name = experiment['file_name']

        g_train = rdflib.Graph()
        g_train.parse(f'datasets/{dataset_name}_train.owl', format='xml')

        g_noise = rdflib.Graph()
        g_noise.parse(f'datasets/noise/{file_name}.owl', format='xml')

        g_train += g_noise
        g_train.serialize(destination=f'datasets/{file_name}_train.owl', format="xml")