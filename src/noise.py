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

def add_triples_logical(g_no_noise, noise_percentage, disjoint_classes, disjoint_properties):
    max_triples = int(noise_percentage * len(g_no_noise))

    # Clean graph
    g_no_noise = remove_triples(g_no_noise)
    
    # Graphs
    noisy_g_logical = rdflib.Graph()           # holds only corrupted triples
    new_g_logical = copy_graph(g_no_noise)     # copy of the new graph

    triples_list = list(g_no_noise)
    all_classes = set(g_no_noise.objects(None, rdflib.RDF.type))

    # Precompute
    class_to_instances = build_class_index(g_no_noise)
    obj_classes = build_object_class_map(g_no_noise)
    alt_classes = {c: list(all_classes - {c}) for c in all_classes}

    # Namespace for fake individuals
    EX = Namespace("http://example.org/fake/")

    # -------------------------------------------------------
    # STEP 1 — Build all logically valid corruptions
    # -------------------------------------------------------
    logical_candidates = set()

    for s, p, o in triples_list:

        # --- TYPE corruption ---
        if p == rdflib.RDF.type:
            if o in disjoint_classes:
                for dc in disjoint_classes[o]:
                    logical_candidates.add((s, p, dc))
            else:
                for alt in alt_classes[o]:
                    logical_candidates.add((s, p, alt))

        # --- PROPERTY corruption ---
        elif p in disjoint_properties:
            for dp in disjoint_properties[p]:
                logical_candidates.add((s, dp, o))

        # --- OBJECT corruption ---
        else:
            for c in obj_classes.get(o, []):
                if c in disjoint_classes:
                    for new_c in disjoint_classes[c]:
                        for inst in class_to_instances.get(new_c, []):
                            logical_candidates.add((s, p, inst))

    # Remove triples already in graph
    logical_candidates = [t for t in logical_candidates if t not in g_no_noise]

    # Shuffle for randomness
    random.shuffle(logical_candidates)

    # How many do we have?
    num_logical = len(logical_candidates)

    # -------------------------------------------------------
    # STEP 2 — Use as many logical corruptions as possible
    # -------------------------------------------------------
    selected = []

    if num_logical >= max_triples:
        selected = logical_candidates[:max_triples]

    else:
        # Use all available logical ones
        selected = list(logical_candidates)

        # ---------------------------------------------------
        # STEP 3 — Generate fake individuals for the rest
        # ---------------------------------------------------
        needed = max_triples - num_logical
        fake_count_1 = 0
        fake_count_2 = 1

        while needed > 0:

            if random.choice([True, False]) and len(disjoint_classes) > 0:
                # ----------------------------------
                # Fake individuals + disjoint class
                # ----------------------------------
                dc = random.choice(list(disjoint_classes.keys()))
                class_list = disjoint_classes[dc]
                if not class_list:
                    continue
                chosen_dc = random.choice(class_list)

                fake_ind_1 = EX[f"Fake{fake_count_1}"]
                fake_ind_2 = EX[f"Fake{fake_count_2}"]
                fake_count_1 += 2
                fake_count_2 += 2

                corrupted_1 = (fake_ind_1, rdflib.RDF.type, chosen_dc)
                corrupted_2 = (fake_ind_2, rdflib.RDF.type, chosen_dc)

                # Add if unique
                if corrupted_1 not in selected and corrupted_1 not in g_no_noise:
                    selected.append(corrupted_1)
                    needed -= 1
                    if needed == 0:
                        break

                if corrupted_2 not in selected and corrupted_2 not in g_no_noise:
                    selected.append(corrupted_2)
                    needed -= 1
                    if needed == 0:
                        break

            else:
                # ----------------------------------
                # Fake individuals + disjoint property
                # ----------------------------------
                if len(disjoint_properties) == 0:
                    continue

                # Pick a random property group
                base_prop = random.choice(list(disjoint_properties.keys()))
                prop_list = disjoint_properties[base_prop]
                if not prop_list:
                    continue

                chosen_prop = random.choice(prop_list)

                fake_ind_1 = EX[f"Fake{fake_count_1}"]
                fake_ind_2 = EX[f"Fake{fake_count_2}"]
                fake_count_1 += 2
                fake_count_2 += 2

                # Build corrupted triple: fake1 -prop-> fake2
                corrupted = (fake_ind_1, chosen_prop, fake_ind_2)

                if corrupted not in selected and corrupted not in g_no_noise:
                    selected.append(corrupted)
                    needed -= 1

    # -------------------------------------------------------
    # STEP 4 — Add selected corruptions to output graphs
    # -------------------------------------------------------
    for t in selected:
        noisy_g_logical.add(t)
        new_g_logical.add(t)

    return noisy_g_logical, new_g_logical