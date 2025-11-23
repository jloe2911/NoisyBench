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

    # Graphs
    noisy_g_random = rdflib.Graph()           # holds only corrupted triples
    new_g_random = copy_graph(g_no_noise)     # copy of the new graph

    subjects = list(set(g_no_noise.subjects()))
    objects = list(set(g_no_noise.objects()))
    triples_list = list(g_no_noise)

    # Select triples to corrupt 
    triples_to_corrupt = random.sample(triples_list, max_triples)

    for triple in triples_to_corrupt:
        s, p, o = triple

        choice = random.choice(['s', 'o'])
        if choice == 's':
            new_s = random.choice(subjects)
            corrupted_triple = (new_s, p, o)
        else:
            new_o = random.choice(objects)
            corrupted_triple = (s, p, new_o)

        # Make sure corruption actually changes the triple and isn't already in the graph
        if corrupted_triple != triple and corrupted_triple not in g_no_noise:
            new_g_random.add(corrupted_triple)
            noisy_g_random.add(corrupted_triple)

    return noisy_g_random, new_g_random

def train_gnn(g, nodes, device, relations, seed=42, epochs=500):
    data = get_data(g)[0]
    data = split_edges(data, test_ratio=0.2, val_ratio=0.1, seed=seed)
    model = GNN(device, len(nodes), len(relations))
    for epoch in range(epochs + 1):
        epoch_seed = seed + epoch
        loss = model._train(data.to(device), epoch_seed=epoch_seed)
    return model, data

def add_triples_gnn(model, g_no_noise, data, nodes_dict_rev, relations_dict_rev, device, noise_percentage):
    max_triples = int((noise_percentage * len(g_no_noise)) / len(relations_dict_rev))
    
    # Graphs
    noisy_g_gnn = rdflib.Graph()           # holds only corrupted triples
    new_g_gnn = copy_graph(g_no_noise)     # copy of the new graph
    
    # Encode once
    output = model.encode(data.to(device))
    scores = torch.matmul(output, output.T)
    output_norm = torch.norm(output, dim=1, keepdim=True)
    scores_norm = scores / (output_norm * output_norm.T)
    
    for key in relations_dict_rev:
        mask = data.edge_type == key
        edge_index = torch.tensor([data.edge_index[0, mask].tolist(), data.edge_index[1, mask].tolist()])

        # Set scores of existing edges of this relation to 1 to exclude
        scores_norm[edge_index[0, :], edge_index[1, :]] = 1

        _, topk_indices = torch.topk(scores_norm.flatten(), max_triples * 2, largest=False)
        row_idx, col_idx = topk_indices // scores_norm.size(1), topk_indices % scores_norm.size(1)

        # For directed graphs, you may want to keep all pairs
        valid_mask = row_idx < col_idx
        row_idx, col_idx = row_idx[valid_mask], col_idx[valid_mask]

        for r, c in zip(row_idx, col_idx):
            s = nodes_dict_rev[r.item()]
            o = nodes_dict_rev[c.item()]
            p = relations_dict_rev[key]
            corrupted_triple = (s, rdflib.URIRef(p), o)  # Directly create a triple from the candidate pair

            if corrupted_triple not in g_no_noise:
                noisy_g_gnn.add(corrupted_triple)
                new_g_gnn.add(corrupted_triple)

    return noisy_g_gnn, new_g_gnn

def build_class_index(g):
    """Return dict: class_uri -> set(instances)"""
    ci = {}
    for s, p, o in g.triples((None, RDF.type, None)):
        ci.setdefault(o, set()).add(s)
    return ci

def build_domain_range_maps(g):
    domain_map = defaultdict(set)
    range_map = defaultdict(set)

    for p, o in g.subject_objects(rdflib.RDFS.domain):
        domain_map[p].add(o)

    for p, o in g.subject_objects(rdflib.RDFS.range):
        range_map[p].add(o)

    return dict(domain_map), dict(range_map)

def add_triples_logical(
        g_no_noise,
        noise_percentage,
        disjoint_classes,
        disjoint_properties,
        domain_map,
        range_map
):
    """
    Fast logical noise generator:
    - disjoint class violations
    - disjoint property violations
    - domain violations
    - range violations
    - fake nodes if needed
    """

    # -------------------------------------------------------
    # PREP
    # -------------------------------------------------------
    n_target = int(noise_percentage * len(g_no_noise))

    # Clean original
    g_no_noise = remove_triples(g_no_noise)
    new_graph = copy_graph(g_no_noise)
    noise_graph = rdflib.Graph()

    triples = list(g_no_noise)

    ## Pre-compute structures
    class_to_instances = build_class_index(g_no_noise)

    # Reverse index: instance → classes
    inst_classes = defaultdict(set)
    for cls, insts in class_to_instances.items():
        for inst in insts:
            inst_classes[inst].add(cls)

    EX = Namespace("http://example.org/fake/")

    # -------------------------------------------------------
    # FAST HELPERS
    # -------------------------------------------------------
    def violating_instance(target_class):
        """Return an existing instance of a class incompatible with target_class."""
        if target_class in disjoint_classes:
            possible = []
            for dc in disjoint_classes[target_class]:
                possible.extend(class_to_instances.get(dc, []))
            return random.choice(possible) if possible else None
        return None

    # -------------------------------------------------------
    # STEP 1 – Generate violations on the fly (FAST)
    # -------------------------------------------------------
    selected = []

    for s, p, o in triples:

        # --- TYPE (rdf:type) → DISJOINT CLASS VIOLATION ---
        if p == rdflib.RDF.type:
            cls = o
            if cls in disjoint_classes and disjoint_classes[cls]:
                bad_cls = random.choice(disjoint_classes[cls])
                selected.append((s, rdflib.RDF.type, bad_cls))
        
        # --- PROPERTY → DISJOINT PROPERTY VIOLATION ---
        elif p in disjoint_properties and disjoint_properties[p]:
            bad_prop = random.choice(disjoint_properties[p])
            selected.append((s, bad_prop, o))
        
        # --- DOMAIN VIOLATION ---
        elif p in domain_map:
            expected_classes = domain_map[p]
            # pick an instance violating domain
            violating = None
            for c in expected_classes:
                violating = violating_instance(c)
                if violating:
                    break
            if violating:
                selected.append((violating, p, o))

        # --- RANGE VIOLATION ---
        elif p in range_map:
            expected_classes = range_map[p]
            violating = None
            for c in expected_classes:
                violating = violating_instance(c)
                if violating:
                    break
            if violating:
                selected.append((s, p, violating))

    # Remove any triple already present
    selected = [t for t in selected if t not in g_no_noise]

    # Shuffle
    random.shuffle(selected)

    # -------------------------------------------------------
    # STEP 2 – Trim or supplement with fake nodes
    # -------------------------------------------------------
    if len(selected) >= n_target:
        selected = selected[:n_target]
    else:
        # Need more noise → add fake individuals
        needed = n_target - len(selected)

        fake_id = 0
        while needed > 0:

            # randomly choose mode
            mode = random.choice(["class", "prop"])

            # --- Fake disjoint class noise ---
            if mode == "class" and len(disjoint_classes) > 0:
                base = random.choice(list(disjoint_classes.keys()))
                options = disjoint_classes[base]
                if options:
                    cls = random.choice(options)
                    fake = EX[f"Fake{fake_id}"]
                    fake_id += 1
                    t = (fake, rdflib.RDF.type, cls)
                    if t not in g_no_noise:
                        selected.append(t)
                        needed -= 1
                        continue

            # --- Fake disjoint property noise ---
            if mode == "prop" and len(disjoint_properties) > 0:
                base = random.choice(list(disjoint_properties.keys()))
                plist = disjoint_properties[base]
                if plist:
                    p2 = random.choice(plist)
                    fake1 = EX[f"Fake{fake_id}"]; fake_id += 1
                    fake2 = EX[f"Fake{fake_id}"]; fake_id += 1
                    t = (fake1, p2, fake2)
                    if t not in g_no_noise:
                        selected.append(t)
                        needed -= 1

    # -------------------------------------------------------
    # STEP 3 – Add to graphs
    # -------------------------------------------------------
    for t in selected:
        noise_graph.add(t)
        new_graph.add(t)

    return noise_graph, new_graph