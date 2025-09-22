import os
import logging
from glob import glob
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import rdflib
from rdflib import URIRef, OWL, RDF, BNode
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

def test_val_split(df: pd.DataFrame, test_size: float, stratify_col: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[stratify_col])

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

def merge_nt_files_parallel(nt_files, output_file: str):
    """Merge multiple RDF files in parallel and save in correct format."""
    merged_graph = rdflib.Graph()
    if len(nt_files) == 0:
        logging.warning(f"No files to merge for {output_file}")
    else:
        with ThreadPoolExecutor() as executor:
            for g in tqdm(executor.map(parse_file, nt_files), total=len(nt_files), desc="Merging inference files"):
                merged_graph += g

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Serialize using extension-based format
    merged_graph.serialize(destination=output_file, format=get_format_from_extension(output_file))
    logging.info(f"Merged file created at {output_file}")

# --------------------------
# Graph Utilities
# --------------------------
def remove_bnodes(graph: rdflib.Graph) -> rdflib.Graph:
    new_graph = rdflib.Graph()
    for s, p, o in graph:
        if isinstance(s, BNode) or isinstance(p, BNode) or isinstance(o, BNode):
            continue
        new_graph.add((s, p, o))
    return new_graph

def get_entities(graph: rdflib.Graph):
    classes, individuals, relations = set(), set(), set()
    for s, p, o in graph:
        if (s, RDF.type, OWL.Class) in graph: classes.add(s)
        if (o, RDF.type, OWL.Class) in graph: classes.add(o)
        if (s, RDF.type, OWL.NamedIndividual) in graph: individuals.add(s)
        if (o, RDF.type, OWL.NamedIndividual) in graph: individuals.add(o)
        if isinstance(p, URIRef): relations.add(p)
    return classes, relations, individuals

def remove_missing_entities(train_graph: rdflib.Graph, target_graph: rdflib.Graph) -> rdflib.Graph:
    train_classes, train_relations, train_individuals = get_entities(train_graph)
    classes, relations, individuals = get_entities(target_graph)
    missing_classes = classes - train_classes
    missing_relations = relations - train_relations
    missing_individuals = individuals - train_individuals
    new_graph = rdflib.Graph()
    for s, p, o in target_graph:
        if s in missing_classes or o in missing_classes: continue
        if p in missing_relations: continue
        if s in missing_individuals or o in missing_individuals: continue
        new_graph.add((s, p, o))
    return new_graph

# --------------------------
# Main Pipeline Function
# --------------------------
def build_rdf_datasets(dataset_name: str, test_size: float = 0.5, seed: int = 1):
    logging.info(f"Starting RDF dataset pipeline for {dataset_name}")

    # Ensure datasets folder exists
    os.makedirs('datasets', exist_ok=True)

    # --------------------------
    # Load ontology
    # --------------------------
    ontology_path = os.path.join('ontologies', f"{dataset_name}.owl")
    get_ontology(ontology_path).load()
    
    # --------------------------
    # Prepare input/inference dataframe
    # --------------------------
    files_df = get_files_df(
        f'datasets/{dataset_name}_input_graphs_filtered_1hop/',
        f'datasets/{dataset_name}_inferred_graphs_filtered/'
    )
    
    graph_counts = files_df['graph_type'].value_counts()
    files_df = files_df[files_df['graph_type'].isin(graph_counts[graph_counts > 1].index)]

    # --------------------------
    # Test/Val Split
    # --------------------------
    rdf_data_test, rdf_data_val = test_val_split(files_df, test_size=test_size, stratify_col="graph_type", seed=seed)

    # --------------------------
    # Merge inference files (parallel)
    # --------------------------
    test_file = os.path.join('datasets', f"{dataset_name}_test_complete.ttl")
    val_file = os.path.join('datasets', f"{dataset_name}_val_complete.ttl")
    merge_nt_files_parallel(rdf_data_test['inference_file'], test_file)
    merge_nt_files_parallel(rdf_data_val['inference_file'], val_file)

    # --------------------------
    # Load graphs
    # --------------------------
    G_train = rdflib.Graph().parse(
        os.path.join('ontologies', f"{dataset_name}.owl"), 
        format=get_rdf_format(os.path.join('ontologies', f"{dataset_name}.owl"))
    )
    G_tbox = rdflib.Graph().parse(
        os.path.join('ontologies', f"{dataset_name}_TBOX.owl"),
        format=get_rdf_format(os.path.join('ontologies', f"{dataset_name}_TBOX.owl"))
    )
    G_test = rdflib.Graph().parse(test_file, format=get_rdf_format(test_file))
    G_val = rdflib.Graph().parse(val_file, format=get_rdf_format(val_file))

    # --------------------------
    # Add OWL.NamedIndividual triples
    # --------------------------
    for triple in set(G_train.triples((None, None, OWL.NamedIndividual))):
        G_test.add(triple)
        G_val.add(triple)

    # --------------------------
    # Add TBOX
    # --------------------------
    G_test += G_tbox
    G_val += G_tbox

    # --------------------------
    # Remove BNodes
    # --------------------------
    G_train, G_test, G_val = map(remove_bnodes, [G_train, G_test, G_val])

    # --------------------------
    # Remove missing entities
    # --------------------------
    G_test = remove_missing_entities(G_train, G_test)
    G_val = remove_missing_entities(G_train, G_val)

    # --------------------------
    # Serialize final graphs
    # --------------------------
    train_file = os.path.join('datasets', f"{dataset_name}_train.owl")
    final_test_file = os.path.join('datasets', f"{dataset_name}_test.owl")
    final_val_file = os.path.join('datasets', f"{dataset_name}_val.owl")

    G_train.serialize(train_file, format="xml")  
    G_test.serialize(final_test_file, format="xml")
    G_val.serialize(final_val_file, format="xml")

    logging.info(f"Pipeline completed for {dataset_name} - Triples: Train={len(G_train)}, Test={len(G_test)}, Val={len(G_val)}")

    return train_file, final_test_file, final_val_file