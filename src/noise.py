import random
import rdflib
from rdflib import Literal, RDF, OWL, Namespace, URIRef
from collections import defaultdict
import torch

from src.utils import copy_graph
from src.gnn import get_data, split_edges, GNN

def get_disjoint_classes(g):
    """
    Extract disjoint classes from an ontology and represent them as a dictionary.

    Args:
        ontology_path (str): Path to the ontology file.

    Returns:
        dict: Dictionary where keys are classes and values are lists of disjoint classes.
    """

    # Find disjoint classes
    disjoint_classes = {}
    for s, p, o in g.triples((None, OWL.disjointWith, None)):
        if (s, RDF.type, OWL.Class) in g and (o, RDF.type, OWL.Class) in g:
            disjoint_classes.setdefault(s, []).append(o)
            disjoint_classes.setdefault(o, []).append(s)  # To ensure symmetry

    return {k: [v for v in values] for k, values in disjoint_classes.items()}

def get_disjoint_properties(g):
    """
    Extract disjoint properties from an ontology and represent them as a dictionary.

    Args:
        ontology_path (str): Path to the ontology file.

    Returns:
        dict: Dictionary where keys are properties and values are lists of disjoint properties.
    """

    # Find disjoint properties
    disjoint_properties = {}
    for s, p, o in g.triples((None, OWL.propertyDisjointWith, None)):
        disjoint_properties.setdefault(s, []).append(o)
        disjoint_properties.setdefault(o, []).append(s)  # Ensure symmetry

    return {k: [v for v in values] for k, values in disjoint_properties.items()}

def get_possible_predicates(g_no_noise):

    individual_predicates = set()
    
    for subj, pred, obj in g_no_noise:
        if (subj, RDF.type, OWL.NamedIndividual) in g_no_noise and \
           (obj, RDF.type, OWL.NamedIndividual) in g_no_noise and \
           not isinstance(obj, Literal):
            individual_predicates.add(pred)
    
    possible_predicates = [str(pred) for pred in individual_predicates if not str(pred).startswith('http://www.w3.org/')]
    
    return possible_predicates

def remove_triples(graph):
    
    namespace_to_filter = Namespace("http://www.w3.org/2002/07/owl")
    namespace_to_filter2 = Namespace("http://www.w3.org/2000/01/rdf-schema")

    # Find triples to remove
    triples_to_remove = []
    for s, p, o in graph:
        if str(o).startswith(str(namespace_to_filter)) or str(o).startswith(str(namespace_to_filter2)):
            triples_to_remove.append((s, p, o))

    # Remove the triples
    for triple in triples_to_remove:
        graph.remove(triple)

    return graph

def find_instances(graph, class_uri):
    """
    Find all instances of a given class in the graph.
    """
    return {s for s, _, o in graph.triples((None, rdflib.RDF.type, class_uri))}

def add_triples_random(g_no_noise, noise_percentage):
    max_triples = int(noise_percentage * len(g_no_noise)) 

    noisy_g_random = rdflib.Graph()
    new_g_random = copy_graph(g_no_noise)
    num_triples = 0

    subjects = list(set(g_no_noise.subjects()))
    objects = list(set(g_no_noise.objects()))
    triples_list = list(g_no_noise)

    while num_triples < max_triples:
        triple = random.choice(triples_list)
        s, p, o = triple

        if random.choice([True, False]):  
            new_s = random.choice(subjects)
            corrupted_triple = (new_s, p, o)
        else:  
            new_o = random.choice(objects)
            corrupted_triple = (s, p, new_o)

        if corrupted_triple not in g_no_noise:
            noisy_g_random.add(corrupted_triple)
            new_g_random.add(corrupted_triple)
            num_triples += 1
    return noisy_g_random, new_g_random

def train_gnn(g, nodes, device, relations, epochs=100):
    data = get_data(g)[0]
    data = split_edges(data)
    model = GNN(device, len(nodes), len(relations))    
    for _ in range(epochs+1):
        loss = model._train(data.to(device))
    return model, data

def add_triples_gnn(model, g, data, nodes_dict_rev, relations_dict_rev, device, noise_percentage):
    max_triples = int((noise_percentage * len(g)) / len(relations_dict_rev))
    noisy_g = rdflib.Graph()
    new_g = copy_graph(g)
    
    for key in relations_dict_rev:
        mask = data.edge_type == key
        edge_index = torch.tensor([data.edge_index[0, mask].tolist(), data.edge_index[1, mask].tolist()])
        edge_type = data.edge_type[mask]

        output = model.model.encode(edge_index.to(device), edge_type.to(device))
        scores = torch.matmul(output, output.T)
        output_norm = torch.norm(output, dim=1, keepdim=True)
        scores_norm = scores / (output_norm * output_norm.T)
        scores_norm[edge_index[0, :], edge_index[1, :]] = 1

        _, topk_indices = torch.topk(scores_norm.flatten(), max_triples * 2, largest=False)
        row_idx, col_idx = topk_indices // scores_norm.size(1), topk_indices % scores_norm.size(1)
        valid_mask = row_idx < col_idx
        row_idx, col_idx = row_idx[valid_mask], col_idx[valid_mask]

        for r, c in zip(row_idx, col_idx):
            s = nodes_dict_rev[r.item()]
            o = nodes_dict_rev[c.item()]
            p = relations_dict_rev[key]
            existing_triples = list(g.triples((None, URIRef(p), None)))
            if existing_triples:
                triple = random.choice(existing_triples)
                corrupted = (s, triple[1], triple[2]) if random.choice([True, False]) else (triple[0], triple[1], o)
                if corrupted not in g:
                    noisy_g.add(corrupted)
                    new_g.add(corrupted)
    
    return noisy_g, new_g

def build_class_index(graph):
    class_to_instances = defaultdict(set)
    for s, _, o in graph.triples((None, rdflib.RDF.type, None)):
        class_to_instances[o].add(s)
    return class_to_instances

def build_object_class_map(graph):
    obj_classes = defaultdict(set)
    for s, _, o in graph.triples((None, rdflib.RDF.type, None)):
        obj_classes[s].add(o)
    return obj_classes

def add_triples_logical(graph, noise_percentage, disjoint_classes, disjoint_properties):
    graph = remove_triples(graph)
    max_triples = int(noise_percentage * len(graph))

    noisy_graph = rdflib.Graph()
    new_graph = copy_graph(graph)

    triples_list = list(graph)
    all_classes = set(graph.objects(None, rdflib.RDF.type))

    # Precompute
    class_to_instances = build_class_index(graph)
    obj_classes = build_object_class_map(graph)
    alt_classes = {c: list(all_classes - {c}) for c in all_classes}

    num_triples = 0
    attempts = 0
    while num_triples < max_triples and attempts < max_triples * 50:  
        attempts += 1
        s, p, o = random.choice(triples_list)
        corrupted = None

        if p == rdflib.RDF.type:
            if o in disjoint_classes:
                corrupted = (s, p, random.choice(disjoint_classes[o]))
            else:
                corrupted = (s, p, random.choice(alt_classes[o]))
        elif p in disjoint_properties:
            corrupted = (s, random.choice(disjoint_properties[p]), o)
        else:
            for c in obj_classes.get(o, []):
                if c in disjoint_classes:
                    new_c = random.choice(disjoint_classes[c])
                    candidates = class_to_instances.get(new_c, [])
                    if candidates:
                        corrupted = (s, p, random.choice(list(candidates)))
                        break

        if corrupted and corrupted not in graph and corrupted not in noisy_graph:
            noisy_graph.add(corrupted)
            new_graph.add(corrupted)
            num_triples += 1

    return noisy_graph, new_graph