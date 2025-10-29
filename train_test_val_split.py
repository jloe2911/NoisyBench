import os
import logging
from glob import glob

import pandas as pd
import rdflib
from rdflib import OWL, RDF, RDFS, BNode
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
        if isinstance(s, rdflib.URIRef) and ((s, rdflib.RDF.type, rdflib.OWL.Class) in graph or (s, rdflib.RDF.type, rdflib.RDFS.Class) in graph):
            classes.add(s)
        if isinstance(o, rdflib.URIRef) and ((o, rdflib.RDF.type, rdflib.OWL.Class) in graph or (o, rdflib.RDF.type, rdflib.RDFS.Class) in graph):
            classes.add(o)

        # Individuals = everything that is not a class and is a URI
        if isinstance(s, rdflib.URIRef) and s not in classes:
            individuals.add(s)
        if isinstance(o, rdflib.URIRef) and o not in classes:
            individuals.add(o)

        # Relations
        if isinstance(p, rdflib.URIRef):
            relations.add(p)
    return classes, relations, individuals

# --------------------------
# Entity-Level Split
# --------------------------
def entity_level_split(df: pd.DataFrame, seed: int = 1, test_size: float = 0.1, val_size: float = 0.1):
    """
    Split inferred graphs by INDIVIDUALS (entities), not by graph_type.
    - test_size, val_size: fractions of the TOTAL dataset
    - Returns lists of rdflib.Graph objects for train, val, and test
    """
    logging.info("Performing entity-level split (by individuals)")

    train_graphs, val_graphs, test_graphs = [], [], []

    # Step 0: Include all base input graphs in train
    for g_file in df["input_graph_file"].dropna().unique():
        try:
            g = parse_file(g_file)
            train_graphs.append(g)
        except Exception as e:
            logging.error(f"Failed to parse base graph {g_file}: {e}")

    # Step 1: Collect all individuals from inferred graphs
    all_individuals = set()
    graph_to_inds = {}
    for _, row in df.iterrows():
        inferred_file = row["inference_file"]
        if pd.isna(inferred_file) or not os.path.exists(inferred_file):
            continue
        try:
            g = parse_file(inferred_file)
            _, _, inds = get_entities(g)
            graph_to_inds[inferred_file] = inds
            all_individuals |= inds
        except Exception as e:
            logging.error(f"Failed to parse inferred graph {inferred_file}: {e}")

    all_individuals = list(all_individuals)
    n_total = len(all_individuals)

    if n_total == 0:
        logging.warning("No individuals found — putting everything into train split.")
        return train_graphs, [], []

    logging.info(f"Collected {n_total} unique individuals across all inferred graphs")

    # Step 2: Split individuals into train/val/test
    assert 0.0 < test_size < 1.0, f"test_size must be between 0 and 1 (got {test_size})"
    assert 0.0 <= val_size < 1.0, f"val_size must be between 0 and 1 (got {val_size})"

    # Split test
    train_val_inds, test_inds = train_test_split(
        all_individuals, test_size=test_size, random_state=seed
    )

    # Split val from remaining
    if val_size > 0:
        val_size_adjusted = val_size / (1.0 - test_size)
        train_inds, val_inds = train_test_split(
            train_val_inds, test_size=val_size_adjusted, random_state=seed
        )
    else:
        train_inds, val_inds = train_val_inds, []

    train_inds, val_inds, test_inds = set(train_inds), set(val_inds), set(test_inds)
    logging.info(
        f"Individuals split — Train: {len(train_inds)}, Val: {len(val_inds)}, Test: {len(test_inds)}"
    )

    # Step 3: Filter triples into appropriate splits
    def filter_graph_by_inds(graph_file, train_inds, val_inds, test_inds):
        g = parse_file(graph_file)
        g_train, g_val, g_test = rdflib.Graph(), rdflib.Graph(), rdflib.Graph()

        for s, p, o in g:
            assigned = False
            if s in train_inds or o in train_inds:
                g_train.add((s, p, o)); assigned = True
            if s in val_inds or o in val_inds:
                g_val.add((s, p, o)); assigned = True
            if s in test_inds or o in test_inds:
                g_test.add((s, p, o)); assigned = True
            if not assigned:
                # fallback: keep unknown individuals in training
                g_train.add((s, p, o))
        return g_train, g_val, g_test

    for inferred_file in graph_to_inds.keys():
        g_train, g_val, g_test = filter_graph_by_inds(inferred_file, train_inds, val_inds, test_inds)
        if len(g_train) > 0:
            train_graphs.append(g_train)
        if len(g_val) > 0:
            val_graphs.append(g_val)
        if len(g_test) > 0:
            test_graphs.append(g_test)

    logging.info(
        f"Split results — Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}, Test graphs: {len(test_graphs)}"
    )

    return train_graphs, val_graphs, test_graphs

# --------------------------
# Main Pipeline Function
# --------------------------
def build_rdf_datasets(dataset_name: str, test_size: float = 0.1, val_size: float = 0.1, seed: int = 1):
    """
    Main pipeline for building RDF dataset splits.
    - Loads ontology and TBOX
    - Performs entity-level split
    - Merges and serializes train/val/test graphs
    """
    logging.info(f"Starting RDF dataset pipeline for {dataset_name}")

    os.makedirs('datasets', exist_ok=True)

    # Load ontology 
    ontology_path = os.path.join('ontologies', f"{dataset_name}.owl")
    if os.path.exists(ontology_path):
        try:
            get_ontology(ontology_path).load()
            logging.info(f"Ontology loaded from {ontology_path}")
        except Exception as e:
            logging.warning(f"Failed to load ontology: {e}")
    else:
        logging.warning(f"No ontology found at {ontology_path}")

    # Prepare input/inference dataframe
    input_folder = f'datasets/{dataset_name}_input_graphs_filtered_1hop/'
    inference_folder = f'datasets/{dataset_name}_inferred_graphs_filtered/'

    files_df = get_files_df(input_folder, inference_folder)
    if files_df.empty:
        logging.error("No input/inference file pairs found — aborting pipeline.")
        return None, None, None
    
    # Entity-level split
    train_graphs, val_graphs, test_graphs = entity_level_split(
        files_df, seed=seed, test_size=test_size, val_size=val_size
    )

    # Merge graphs efficiently
    def merge_graph_list(graph_list):
        merged = rdflib.Graph()
        for g in graph_list:
            for triple in g:
                merged.add(triple)
        return merged

    G_train = merge_graph_list(train_graphs)
    G_val = merge_graph_list(val_graphs)
    G_test = merge_graph_list(test_graphs)

    # Load TBOX S
    tbox_path = os.path.join('ontologies', f"{dataset_name}_TBOX.owl")
    if os.path.exists(tbox_path):
        try:
            rdf_format = get_rdf_format(tbox_path)
            G_tbox = rdflib.Graph().parse(tbox_path, format=rdf_format)
            G_train += G_tbox
            G_val += G_tbox
            G_test += G_tbox
            logging.info(f"TBOX loaded and added from {tbox_path}")
        except Exception as e:
            logging.warning(f"Failed to parse TBOX ontology: {e}")
    else:
        logging.warning(f"No TBOX file found for {dataset_name}")

    # Serialize final graphs
    final_train_file = os.path.join('datasets', f"{dataset_name}_train.owl")
    final_val_file   = os.path.join('datasets', f"{dataset_name}_val.owl")
    final_test_file  = os.path.join('datasets', f"{dataset_name}_test.owl")

    for graph, path in [(G_train, final_train_file), (G_val, final_val_file), (G_test, final_test_file)]:
        try:
            graph.serialize(path, format="xml")
        except Exception as e:
            logging.error(f"Failed to serialize {path}: {e}")

    logging.info(
        f"Pipeline completed for {dataset_name} — Triples: "
        f"Train={len(G_train)}, Val={len(G_val)}, Test={len(G_test)}"
    )

    return final_train_file, final_val_file, final_test_file