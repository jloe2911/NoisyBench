from owlready2 import get_ontology
import rdflib
from rdflib import URIRef
from tqdm import tqdm
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.utils import get_individuals

def load_graphs(dataset_name: str):
    """Load ontology, base graph, and TBox graph."""
    ontology = get_ontology(f'ontologies/{dataset_name}.owl').load()

    g = rdflib.Graph()
    g.parse(f'ontologies/{dataset_name}.owl')
    print(f'# Triples: {len(g)}')

    g_tbox = rdflib.Graph()
    g_tbox.parse(f'ontologies/{dataset_name}_TBOX.owl')
    print(f'# Triples (TBox): {len(g_tbox)}')

    return ontology, g, g_tbox

def get_classes_for_individual(individual):
    """Return class labels of an individual as a single string."""
    classes = [cls.name for cls in individual.is_a]
    return "_".join(classes) if classes else "Unknown"

def build_description_graph(subject, g, individuals, hops=1, include_tbox=False, g_tbox=None):
    """Construct description graph for a given subject with N hops and optional TBox."""
    describe_graph = rdflib.Graph()

    # First hop
    for triple in g.triples((None, None, URIRef(subject.iri))):
        describe_graph.add(triple)
    for triple in g.triples((URIRef(subject.iri), None, None)):
        describe_graph.add(triple)

    # Second hop (if required)
    if hops >= 2:
        for triple in list(describe_graph.triples((None, None, None))):
            if triple[0] in individuals and triple[2] in individuals:  # restrict to individuals
                for neighbor in (triple[0], triple[2]):
                    for second_hop_triple in g.triples((neighbor, None, None)):
                        describe_graph.add(second_hop_triple)
                    for second_hop_triple in g.triples((None, None, neighbor)):
                        describe_graph.add(second_hop_triple)

    # Add TBox if needed
    if include_tbox and g_tbox:
        describe_graph += g_tbox

    return describe_graph

def generate_graphs(dataset_name: str):
    ontology, g, g_tbox = load_graphs(dataset_name)
    _, individuals, _ = get_individuals(g)

    subject_resources = sorted(list(ontology.individuals()), key=lambda x: x.name)
    logging.info(f'# Subject-Resources: {len(subject_resources)}')

    # Define output configurations
    configs = [
        {"hops": 2, "include_tbox": True, "out_dir": f"datasets/{dataset_name}_input_graphs"},
        {"hops": 1, "include_tbox": False, "out_dir": f"datasets/{dataset_name}_input_graphs_filtered_1hop"},
        {"hops": 2, "include_tbox": False, "out_dir": f"datasets/{dataset_name}_input_graphs_filtered"},
    ]

    for cfg in configs:
        os.makedirs(cfg["out_dir"], exist_ok=True)
        logging.info(f"Processing: hops={cfg['hops']}, TBox={cfg['include_tbox']}, out_dir={cfg['out_dir']}")

        for subject in tqdm(subject_resources):
            describe_graph = build_description_graph(
                subject, g, individuals,
                hops=cfg["hops"], include_tbox=cfg["include_tbox"], g_tbox=g_tbox
            )

            classes = get_classes_for_individual(subject)
            filename = f"{classes}_{subject.name}.ttl"
            describe_graph.serialize(destination=os.path.join(cfg["out_dir"], filename), format="ttl")