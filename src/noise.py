import rdflib
from rdflib import Literal, RDF, OWL, Namespace
from owlready2 import AllDisjoint

from src.utils import *
from src.gnn import *
from src.sparql_queries import *

from rdflib import Graph, Namespace, RDF, OWL

def get_disjoint_classes(g):
    """
    Extract disjoint classes from an ontology and represent them as a dictionary.

    Args:
        ontology_path (str): Path to the ontology file.

    Returns:
        dict: Dictionary where keys are classes and values are lists of disjoint classes.
    """

    # Define OWL namespace
    OWL_NS = Namespace("http://www.w3.org/2002/07/owl#")

    # Find disjoint classes
    disjoint_classes = {}
    for s, p, o in g.triples((None, OWL_NS.disjointWith, None)):
        if (s, RDF.type, OWL_NS.Class) in g and (o, RDF.type, OWL_NS.Class) in g:
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

    # Define OWL namespace
    OWL_NS = Namespace("http://www.w3.org/2002/07/owl#")

    # Find disjoint properties
    disjoint_properties = {}
    for s, p, o in g.triples((None, OWL_NS.propertyDisjointWith, None)):
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

def add_triples_logical(graph, noise_percentage, disjoint_classes, disjoint_properties):
    """
    Add random corrupted triples to a graph based on disjoint classes and properties.

    Parameters:
    - graph: The original RDF graph.
    - noise_percentage: Percentage of noise to add as corrupted triples.
    - disjoint_classes: A dictionary mapping classes to their disjoint counterparts.
    - disjoint_properties: A dictionary mapping properties to their disjoint counterparts.

    Returns:
    - noisy_graph: A graph containing only the added noisy triples.
    - new_graph: The original graph with noisy triples added.
    """
    # We remove certain triples from the graph (i.e., TBox triples)
    graph = remove_triples(graph)

    # Calculate the number of noisy triples to add
    max_triples = int(noise_percentage * len(graph))  

    noisy_graph = rdflib.Graph()
    new_graph = copy_graph(graph)
    num_triples = 0

    # Precompute sets for efficiency
    triples_list = list(graph)
    all_classes = set(graph.objects(None, rdflib.RDF.type))

    while num_triples < max_triples:
        triple = random.choice(triples_list)
        s, p, o = triple
        # print (f'Triple: {triple}')
        corrupted_triple = None

        if p == rdflib.RDF.type:
            # Handle type triples
            if o in disjoint_classes:
                corrupted_triple = (s, p, random.choice(disjoint_classes[o]))
            else:
                corrupted_triple = (s, p, random.choice(list(all_classes - {o})))
        elif p in disjoint_properties:
            # Handle disjoint properties
            new_p = random.choice(disjoint_properties[p])
            corrupted_triple = (s, new_p, o)
        else:
            # Handle object triples
            o_classes = list(graph.objects(o, rdflib.RDF.type))
            for c in o_classes:
                if c in disjoint_classes:
                    new_c = random.choice(disjoint_classes[c])
                    new_o_candidates = list(find_instances(graph, new_c))
                    if new_o_candidates:
                        new_o = random.choice(new_o_candidates)
                        corrupted_triple = (s, p, new_o)
                        break
                    
        # Add corrupted triple if it's valid and unique
        if corrupted_triple and corrupted_triple not in graph and corrupted_triple not in noisy_graph:
            noisy_graph.add(corrupted_triple)
            new_graph.add(corrupted_triple)
            # print (f'Corrupted Triple: {corrupted_triple}')
            num_triples += 1

    return noisy_graph, new_graph