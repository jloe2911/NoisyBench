import os
import logging
from glob import glob

import pandas as pd
import rdflib
from rdflib import RDF, RDFS, BNode
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
# Split
# --------------------------
def graph_level_random_split(
    inference_folder: str,
    dataset_name: str,
    seed: int = 1,
    test_size: float = 0.1,
    val_size: float = 0.1
):
    """
    GRAPH-LEVEL random split.

    TRAIN = ontology + random subset of inferred graphs
    VAL   = random subset of inferred graphs
    TEST  = random subset of inferred graphs
    """

    logging.info("Performing RANDOM graph-level split")

    train_graphs, val_graphs, test_graphs = [], [], []

    # ---------------------------------------------------
    # Step 1: Add ONTOLOGY to TRAIN
    # ---------------------------------------------------
    ontology_path = f'ontologies/{dataset_name}.owl'
    logging.info(f"Adding ontology to TRAIN: {ontology_path}")

    try:
        g = rdflib.Graph()
        g.parse(ontology_path, format=get_rdf_format(ontology_path))
        train_graphs.append(g)
    except Exception as e:
        logging.error(f"Failed to parse ontology {ontology_path}: {e}")

    # ---------------------------------------------------
    # Step 2: Collect all inferred graph files
    # ---------------------------------------------------
    inferred_files = sorted(glob(os.path.join(inference_folder, "*")))

    if len(inferred_files) == 0:
        logging.error("No inferred files found — aborting.")
        return train_graphs, [], []

    logging.info(f"Found {len(inferred_files)} inferred graphs")

    # ---------------------------------------------------
    # Step 3: Random split inferred graphs
    # ---------------------------------------------------
    train_val_files, test_files = train_test_split(
        inferred_files,
        test_size=test_size,
        random_state=seed
    )

    if val_size > 0:
        val_adj = val_size / (1 - test_size)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_adj,
            random_state=seed
        )
    else:
        train_files, val_files = train_val_files, []

    # ---------------------------------------------------
    # Step 4: Parse inferred graphs into buckets
    # ---------------------------------------------------
    def add_graphs(files, bucket):
        for f in files:
            try:
                g = parse_file(f)
                bucket.append(g)
            except Exception as e:
                logging.error(f"Failed to parse inferred graph {f}: {e}")

    add_graphs(train_files, train_graphs)
    add_graphs(val_files, val_graphs)
    add_graphs(test_files, test_graphs)

    logging.info(
        f"RESULT — Train graphs: {len(train_graphs)}, "
        f"Val graphs: {len(val_graphs)}, "
        f"Test graphs: {len(test_graphs)}"
    )

    return train_graphs, val_graphs, test_graphs

# --------------------------
# Main Pipeline Function
# --------------------------
def build_rdf_datasets(
    dataset_name: str,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 1
):
    """
    Main pipeline for building RDF dataset splits.
    
    TRAIN = ontology + random subset of inferred graphs
    VAL   = random subset of inferred graphs
    TEST  = random subset of inferred graphs
    """

    logging.info(f"Starting RDF dataset pipeline for {dataset_name}")

    os.makedirs('datasets', exist_ok=True)

    # ---------------------------------------------------
    # Load ontology in Owlready (optional, for classes)
    # ---------------------------------------------------
    ontology_path = os.path.join('ontologies', f"{dataset_name}.owl")
    if os.path.exists(ontology_path):
        try:
            get_ontology(ontology_path).load()
            logging.info(f"Ontology loaded from {ontology_path}")
        except Exception as e:
            logging.warning(f"Failed to load ontology with Owlready2: {e}")
    else:
        logging.warning(f"No ontology found at {ontology_path}")

    # ---------------------------------------------------
    # Inferred graphs folder
    # ---------------------------------------------------
    inference_folder = f'datasets/{dataset_name}_inferred_graphs_filtered/'

    # ---------------------------------------------------
    # Perform RANDOM graph-level split
    # ---------------------------------------------------
    train_graphs, val_graphs, test_graphs = graph_level_random_split(
        inference_folder=inference_folder,
        dataset_name=dataset_name,
        seed=seed,
        test_size=test_size,
        val_size=val_size
    )

    # ---------------------------------------------------
    # Merge graphs efficiently
    # ---------------------------------------------------
    def merge_graph_list(graph_list):
        merged = rdflib.Graph()
        for g in graph_list:
            for triple in g:
                merged.add(triple)
        return merged

    G_train = merge_graph_list(train_graphs)
    G_val   = merge_graph_list(val_graphs)
    G_test  = merge_graph_list(test_graphs)

    # ---------------------------------------------------
    # Add TBOX (schema) to all splits
    # ---------------------------------------------------
    tbox_path = os.path.join('ontologies', f"{dataset_name}_TBOX.owl")
    if os.path.exists(tbox_path):
        try:
            rdf_fmt = get_rdf_format(tbox_path)
            G_tbox = rdflib.Graph().parse(tbox_path, format=rdf_fmt)

            G_train += G_tbox
            G_val   += G_tbox
            G_test  += G_tbox

            logging.info(f"TBOX loaded and added from {tbox_path}")
        except Exception as e:
            logging.warning(f"Failed to parse TBOX ontology: {e}")
    else:
        logging.warning(f"No TBOX file found for {dataset_name}")

    # ---------------------------------------------------
    # Serialize final graphs
    # ---------------------------------------------------
    final_train_file = os.path.join('datasets', f"{dataset_name}_train.owl")
    final_val_file   = os.path.join('datasets', f"{dataset_name}_val.owl")
    final_test_file  = os.path.join('datasets', f"{dataset_name}_test.owl")

    for graph, path in [
        (G_train, final_train_file),
        (G_val,   final_val_file),
        (G_test,  final_test_file)
    ]:
        try:
            graph.serialize(path, format="xml")
            logging.info(f"Serialized: {path}")
        except Exception as e:
            logging.error(f"Failed to serialize {path}: {e}")

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------
    logging.info(
        f"Pipeline completed for {dataset_name} — Triples: "
        f"Train={len(G_train)}, Val={len(G_val)}, Test={len(G_test)}"
    )

    return final_train_file, final_val_file, final_test_file