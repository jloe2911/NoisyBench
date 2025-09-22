from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import rdflib
from rdflib import URIRef, RDFS, RDF, OWL, Literal, BNode
from owlready2 import get_ontology
import glob
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def filter_inferred_triples(input_path, inferred_path, filtered_inferred_path, object_properties):  
    g = rdflib.Graph()
    g.parse(input_path)
    g_inferred = rdflib.Graph()
    g_inferred.parse(inferred_path)

    g_inferred_filtered = rdflib.Graph()
    for triple in g_inferred:
        if ((triple[1] in object_properties or triple[1] in {RDFS.subClassOf, RDF.type}) and 
            not isinstance(triple[2], Literal) and 
            triple[0] != OWL.Thing and                            
            triple[2] != OWL.Thing and    
            triple[0] != OWL.Nothing and                            
            triple[2] != OWL.Nothing and                                  
            not isinstance(triple[0], BNode) and  
            not isinstance(triple[2], BNode) and 
            triple[2] != RDFS.Resource and 
            triple not in g):
            g_inferred_filtered.add(triple)
    
    os.makedirs(os.path.dirname(filtered_inferred_path), exist_ok=True)
    g_inferred_filtered.serialize(filtered_inferred_path, format="turtle")
    return f"Filtered: {filtered_inferred_path}"

def process_filtered_graph(input_graph_path, dataset_name, object_properties):
    try:
        # Normalize paths for cross-platform compatibility
        input_graph_path = os.path.normpath(input_graph_path)
        inferred_dir = os.path.normpath(f'datasets/{dataset_name}_inferred_graphs')
        filtered_dir = os.path.normpath(f'datasets/{dataset_name}_inferred_graphs_filtered')

        if input_graph_path.endswith('.xml'):
            file_name = os.path.basename(input_graph_path).replace('.xml', '')
            input_path = os.path.join(f'datasets/{dataset_name}_input_graphs', f'{file_name}.ttl')
            filtered_inferred_path = os.path.join(filtered_dir, f'{file_name}.ttl')
        else:
            file_name = os.path.basename(input_graph_path)
            input_path = os.path.join(f'datasets/{dataset_name}_input_graphs', file_name)
            filtered_inferred_path = os.path.join(filtered_dir, file_name)

        return filter_inferred_triples(input_path, input_graph_path, filtered_inferred_path, object_properties)

    except Exception as e:
        return f"Error processing {input_graph_path}: {e}"

def run_filtering(dataset_name):
    # Directories
    inferred_graph_dir = os.path.normpath(f'datasets/{dataset_name}_inferred_graphs')
    filtered_graph_dir = os.path.normpath(f'datasets/{dataset_name}_inferred_graphs_filtered')
    os.makedirs(filtered_graph_dir, exist_ok=True)

    # Check inferred graph folder
    if not os.path.exists(inferred_graph_dir):
        raise FileNotFoundError(f"Inferred graphs folder not found: {inferred_graph_dir}")

    inferred_graph_paths = sorted(glob.glob(os.path.join(inferred_graph_dir, '*')))
    if not inferred_graph_paths:
        raise FileNotFoundError(f"No files found in {inferred_graph_dir}")

    # Load ontology and extract object properties
    ontology_path = os.path.normpath(f'ontologies/{dataset_name}.owl')
    if not os.path.exists(ontology_path):
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

    ontology = get_ontology(ontology_path).load()
    object_properties = [URIRef(x.iri) for x in ontology.object_properties()]
    logger.info(f"Loaded {len(object_properties)} object properties.")

    # Parallel processing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_filtered_graph, path, dataset_name, object_properties): path
                   for path in inferred_graph_paths}

        for future in tqdm(as_completed(futures), total=len(futures)):
            logger.info(future.result())