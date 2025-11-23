import os
import torch
import rdflib
from rdflib import Namespace, RDF, RDFS, OWL, BNode, URIRef, Literal
from rdflib.collection import Collection

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.gnn import get_data
from src.noise import train_gnn, get_disjoint_classes, get_disjoint_properties, build_domain_range_maps, add_triples_random, add_triples_gnn, add_triples_logical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # Override for consistency

def load_graphs(dataset_name: str):
    g = rdflib.Graph()
    g.parse(f'datasets/{dataset_name}_test.owl', format='xml')
    return g

# ---------- RDF/XML Fix Utilities ----------

def fix_literals_for_rdfxml(g: rdflib.Graph) -> rdflib.Graph:
    """
    Replace literals used in places where IRIs are expected with synthetic IRIs.
    """
    for s, p, o in list(g.triples((None, None, None))):
        if isinstance(o, Literal) and (p in [OWL.someValuesFrom, OWL.allValuesFrom, OWL.hasValue]):
            new_iri = URIRef(f"http://example.com/literal/{str(o)}")
            g.set((s, p, new_iri))
    return g

def fix_blank_nodes_for_rdfxml(g: rdflib.Graph) -> rdflib.Graph:
    """
    Replace blank nodes that RDF/XML cannot serialize with synthetic IRIs.
    """
    def replace_with_iri(node):
        new_iri = URIRef(f"http://example.com/genid/{str(node)[6:20]}")
        # Copy all triples out of the blank node
        for p, o in list(g.predicate_objects(node)):
            g.add((new_iri, p, o))
        for s, p in list(g.subject_predicates(node)):
            g.add((s, p, new_iri))
        g.remove((node, None, None))
        g.remove((None, None, node))
        return new_iri

    # Fix unionOf and intersectionOf
    for s, p, o in list(g.triples((None, OWL.unionOf, None))):
        if isinstance(o, BNode):
            g.set((s, OWL.unionOf, replace_with_iri(o)))
    for s, p, o in list(g.triples((None, OWL.intersectionOf, None))):
        if isinstance(o, BNode):
            g.set((s, OWL.intersectionOf, replace_with_iri(o)))

    # Fix restrictions
    for s, p, o in list(g.triples((None, None, None))):
        if p in [OWL.someValuesFrom, OWL.allValuesFrom, OWL.hasValue] and isinstance(o, BNode):
            replace_with_iri(o)

    # Remove dangling blank nodes
    for s in list(g.subjects()):
        if isinstance(s, BNode) and len(list(g.predicate_objects(s))) == 0:
            g.remove((s, None, None))

    return g

def fix_malformed_disjoint_classes(g):
    to_remove = []

    # AllDisjointClasses form
    for s in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for members in g.objects(s, OWL.members):
            try:
                items = list(Collection(g, members))
            except:
                continue  # malformed list, skip

            if len(items) < 2:
                # mark entire axiom for removal
                to_remove.append(s)

    # Remove malformed AllDisjointClasses axioms
    for ax in to_remove:
        g.remove((ax, None, None))
        g.remove((None, None, ax))

    return g

def fix_disjoint_axioms(g):
    # Fix AllDisjointClasses lists
    g = fix_malformed_disjoint_classes(g)

    # Fix DisjointClasses represented as rdf:List
    for s, p, o in g.triples((None, OWL.disjointWith, None)):
        # disjointWith should point to IRIs or blank nodes, not lists
        # but your noisy generator may produce garbage
        if (o, RDF.first, None) in g:
            # list root detected
            items = []
            try:
                items = list(Collection(g, o))
            except:
                pass
            if len(items) < 2:
                # Remove the broken part
                g.remove((s, OWL.disjointWith, o))

    return g

def sanitize_ontology(graph: rdflib.Graph, base_uri="http://example.com/genid/") -> rdflib.Graph:
    """
    Converts all blank nodes in a graph to synthetic URIs for RDF/XML compatibility.
    """
    bnode_map = {}
    def replace_node(n):
        if isinstance(n, BNode):
            if n not in bnode_map:
                bnode_map[n] = URIRef(f"{base_uri}{str(n)}")
            return bnode_map[n]
        return n

    g_fixed = rdflib.Graph()
    for s, p, o in graph:
        g_fixed.add((replace_node(s), p, replace_node(o)))
    return g_fixed

# ---------- Main Noise Pipeline ----------

def add_noise_to_dataset(dataset_name: str, experiments: dict):
    g = load_graphs(dataset_name)

    nodes, nodes_dict, relations, relations_dict = get_data(g)[1:]
    nodes_dict_rev = {v: k for k, v in nodes_dict.items()}
    relations_dict_rev = {v: k for k, v in relations_dict.items()}

    # Train GNN
    model, data = train_gnn(g, nodes, device, relations)
    os.makedirs("models", exist_ok=True)
    torch.save(model, f'models/RGCN_{dataset_name}')

    # DL Noise
    all_disjoint_classes = get_disjoint_classes(g)
    all_disjoint_properties = get_disjoint_properties(g)
    domain_map, range_map = build_domain_range_maps(g)

    os.makedirs("datasets/noise", exist_ok=True)

    for noise_percentage in [0.25, 0.5, 0.75, 1]:
        # Random
        g = load_graphs(dataset_name)
        noisy_r, new_noisy_r = add_triples_random(g, noise_percentage)
        noisy_r = fix_blank_nodes_for_rdfxml(fix_literals_for_rdfxml(noisy_r))
        noisy_r = fix_malformed_disjoint_classes(noisy_r)
        noisy_r = sanitize_ontology(noisy_r) 
        noisy_r.serialize(f"datasets/noise/{dataset_name}_random_{noise_percentage}.owl", format='xml')
        new_noisy_r = fix_blank_nodes_for_rdfxml(fix_literals_for_rdfxml(new_noisy_r))
        new_noisy_r = fix_malformed_disjoint_classes(new_noisy_r)
        new_noisy_r = sanitize_ontology(new_noisy_r) 
        new_noisy_r.serialize(f"datasets/{dataset_name}_random_{noise_percentage}_test.owl", format='xml')
        logging.info(f"DONE - Random Noise - {noise_percentage}")

        # GNN
        g = load_graphs(dataset_name)
        model = torch.load(f'models/RGCN_{dataset_name}', weights_only=False)
        noisy_g, new_noisy_g = add_triples_gnn(model, g, data, nodes_dict_rev, relations_dict_rev, device, noise_percentage)
        noisy_g = fix_blank_nodes_for_rdfxml(fix_literals_for_rdfxml(noisy_g))
        noisy_g = fix_malformed_disjoint_classes(noisy_g)
        noisy_g = sanitize_ontology(noisy_g) 
        noisy_g.serialize(f"datasets/noise/{dataset_name}_gnn_{noise_percentage}.owl", format='xml')
        new_noisy_g = fix_blank_nodes_for_rdfxml(fix_literals_for_rdfxml(new_noisy_g))
        new_noisy_g = fix_malformed_disjoint_classes(new_noisy_g)
        new_noisy_g = sanitize_ontology(new_noisy_g) 
        new_noisy_g.serialize(f"datasets/{dataset_name}_gnn_{noise_percentage}_test.owl", format='xml')
        logging.info(f"DONE - Statistical Noise - {noise_percentage}")

        # Logical
        g = load_graphs(dataset_name)
        noisy_dl, new_noisy_dl = add_triples_logical(g, noise_percentage, all_disjoint_classes, all_disjoint_properties, domain_map, range_map)
        noisy_dl = fix_blank_nodes_for_rdfxml(fix_literals_for_rdfxml(noisy_dl))
        noisy_dl = fix_malformed_disjoint_classes(noisy_dl)
        noisy_dl = sanitize_ontology(noisy_dl) 
        noisy_dl.serialize(f"datasets/noise/{dataset_name}_logical_{noise_percentage}.owl", format='xml')
        new_noisy_dl = fix_blank_nodes_for_rdfxml(fix_literals_for_rdfxml(new_noisy_dl))
        new_noisy_dl = fix_malformed_disjoint_classes(new_noisy_dl)
        new_noisy_dl = sanitize_ontology(new_noisy_dl) 
        new_noisy_dl.serialize(f"datasets/{dataset_name}_logical_{noise_percentage}_test.owl", format='xml')
        logging.info(f"DONE - Logical Noise - {noise_percentage}")