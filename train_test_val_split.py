import os
import logging
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import rdflib
from rdflib import URIRef, OWL, RDF, RDFS, BNode
from owlready2 import get_ontology
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --------------------------
# Setup logging
# --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------
# Utility Functions
# --------------------------
def get_graph_type(file_path: str) -> str:
    """Extract graph type from filename."""
    name_parts = os.path.basename(file_path).split('_')[:-1]
    return "_".join(name_parts)

def get_files_df(input_folder: str, inference_folder: str) -> pd.DataFrame:
    logging.info(f"Creating dataframe for input/inference pairs")
    rdf_files = []
    for input_graph_path in tqdm(sorted(glob(os.path.join(input_folder, "*"))), desc="Listing input files"):
        inference_path = os.path.join(inference_folder, os.path.basename(input_graph_path))
        if os.path.exists(inference_path):
            rdf_files.append({
                "input_graph_file": input_graph_path,
                "inference_file": inference_path,
                "graph_type": get_graph_type(input_graph_path)
            })
    return pd.DataFrame(rdf_files)

def get_rdf_format(file_path: str) -> str:
    """
    Detects RDF format by checking the first few bytes of an existing file.
    Returns 'turtle' or 'xml'.
    """
    with open(file_path, "rb") as f:
        head = f.read(200).lower()
    if b"<rdf" in head or b"<?xml" in head:
        return "xml"
    else:
        return "turtle"

def get_format_from_extension(file_path: str) -> str:
    """
    Returns RDF format based on file extension.
    Used for serializing files that may not exist yet.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".ttl", ".n3"]:
        return "turtle"
    elif ext in [".rdf", ".owl", ".xml"]:
        return "xml"
    else:
        logging.warning(f"Unknown extension {ext}, defaulting to turtle")
        return "turtle"

# --------------------------
# Merge Functions (Parallel)
# --------------------------
def parse_file(file_path):
    g = rdflib.Graph()
    g.parse(file_path, format=get_rdf_format(file_path))
    return g

# --------------------------
# Graph Utilities
# --------------------------
def get_entities(graph: rdflib.Graph):
    classes, individuals, relations = set(), set(), set()
    for s, p, o in graph:
        # Classes
        if (s, RDF.type, OWL.Class) in graph or (s, RDF.type, RDFS.Class) in graph:
            classes.add(s)
        if (o, RDF.type, OWL.Class) in graph or (o, RDF.type, RDFS.Class) in graph:
            classes.add(o)

        # Individuals: any subject or object with a type that is not a class
        if p == RDF.type and o not in [OWL.Class, RDFS.Class]:
            individuals.add(s)

        # Properties (relations)
        if isinstance(p, URIRef):
            relations.add(p)

    return classes, relations, individuals

# --------------------------
# Entity-Level Split
# --------------------------
def entity_level_split(df: pd.DataFrame, seed: int = 1, test_size: float = 0.2, val_size: float = 0.2):
    """
    Split RDF graphs at the entity level (individuals), not graph level.
    Returns lists of rdflib.Graph objects for train, val, test.
    """
    logging.info("Performing entity-level split")

    # 1. Collect all individuals from all graphs
    all_individuals = set()
    graph_to_individuals = {}
    for idx, row in df.iterrows():
        g = parse_file(row['inference_file'])
        _, _, inds = get_entities(g)
        graph_to_individuals[row['inference_file']] = inds
        all_individuals |= inds

    # 2. Split individuals into train/val/test
    all_individuals = list(all_individuals)
    train_inds, test_inds = train_test_split(all_individuals, test_size=test_size, random_state=seed)
    train_inds, val_inds = train_test_split(train_inds, test_size=val_size / (1 - test_size), random_state=seed)

    train_inds = set(train_inds)
    val_inds = set(val_inds)
    test_inds = set(test_inds)

    logging.info(f"Individuals - Train: {len(train_inds)}, Val: {len(val_inds)}, Test: {len(test_inds)}")

    # 3. Filter graphs by individuals 
    def filter_graph_by_individuals(graph_file, allowed_inds):
        g = parse_file(graph_file)
        new_g = rdflib.Graph()
        for s, p, o in g:
            if (s in allowed_inds) or (o in allowed_inds):
                new_g.add((s, p, o))
        return new_g

    filtered_graphs = {'train': [], 'val': [], 'test': []}
    for row in df.itertuples():
        g_file = row.inference_file
        inds = graph_to_individuals[g_file]
        for split_name, split_inds in [('train', train_inds), ('val', val_inds), ('test', test_inds)]:
            if len(inds & split_inds) > 0:
                fg = filter_graph_by_individuals(g_file, split_inds)
                filtered_graphs[split_name].append(fg)

    return filtered_graphs['train'], filtered_graphs['val'], filtered_graphs['test']

# --------------------------
# Main Pipeline Function
# --------------------------
def build_rdf_datasets(dataset_name: str, test_size: float = 0.2, val_size: float = 0.2, seed: int = 1):
    logging.info(f"Starting RDF dataset pipeline for {dataset_name}")

    os.makedirs('datasets', exist_ok=True)

    # Load ontology
    ontology_path = os.path.join('ontologies', f"{dataset_name}.owl")
    get_ontology(ontology_path).load()

    # Prepare input/inference dataframe
    files_df = get_files_df(
        f'datasets/{dataset_name}_input_graphs_filtered_1hop/',
        f'datasets/{dataset_name}_inferred_graphs_filtered/'
    )

    # Entity-level split
    train_graphs, val_graphs, test_graphs = entity_level_split(files_df, seed=seed, test_size=test_size, val_size=val_size)

    # Merge graphs for each split
    def merge_graph_list(graph_list):
        merged = rdflib.Graph()
        for g in graph_list:
            merged += g
        return merged

    G_train = merge_graph_list(train_graphs)
    G_val = merge_graph_list(val_graphs)
    G_test = merge_graph_list(test_graphs)

    # Load TBOX
    G_tbox = rdflib.Graph().parse(
        os.path.join('ontologies', f"{dataset_name}_TBOX.owl"),
        format=get_rdf_format(os.path.join('ontologies', f"{dataset_name}_TBOX.owl"))
    )

    # Add TBOX to each split
    G_train += G_tbox
    G_val += G_tbox
    G_test += G_tbox

    # Serialize final graphs
    final_train_file = os.path.join('datasets', f"{dataset_name}_train.owl")
    final_val_file   = os.path.join('datasets', f"{dataset_name}_val.owl")
    final_test_file  = os.path.join('datasets', f"{dataset_name}_test.owl")

    G_train.serialize(final_train_file, format="xml")
    G_val.serialize(final_val_file, format="xml")
    G_test.serialize(final_test_file, format="xml")

    logging.info(
        f"Pipeline completed for {dataset_name} - Triples: "
        f"Train={len(G_train)}, Val={len(G_val)}, Test={len(G_test)}"
    )

    return final_train_file, final_val_file, final_test_file