import rdflib
from rdflib import Literal, RDF
from owlready2 import AllDisjoint

from src.utils import *
from src.gnn import *
from src.sparql_queries import *

def get_disjoint_classes(ontology):
    disjoint_classes = []
    for disjoint in ontology.disjoint_classes():
        if isinstance(disjoint, AllDisjoint):
            disjoint_classes.append(disjoint)

    all_disjoint_classes = []
    for disjoint in disjoint_classes:
        all_disjoint_classes.append([cls.name for cls in disjoint.entities])
    return all_disjoint_classes

def get_disjoint_properties(ontology):    
    disjoint_properties = []
    for disjoint in ontology.disjoint_properties():
        if isinstance(disjoint, AllDisjoint):
            disjoint_properties.append(disjoint)

    all_disjoint_properties = []
    for disjoint in disjoint_properties:
        all_disjoint_properties.append([cls.name for cls in disjoint.entities])
    return all_disjoint_properties

def add_noise_disjoint_classes(g_no_noise, max_triples, all_disjoint_classes, uri):
    noisy_g_disjoint = rdflib.Graph()
    _, individual_names, _ = get_individuals(g_no_noise)
    max_combinations = len(individual_names) * sum(len(class_set) * (len(class_set) - 1) // 2 for class_set in all_disjoint_classes)
    if max_triples > max_combinations:
        needed_individuals = (max_triples - max_combinations) // (max_combinations // len(individual_names)) + 1
        individual_names.extend([URIRef(uri + f"I_{i}") for i in range(1, needed_individuals + 1)])
        print('We created new individuals...')

    num_triples = 0
    while num_triples < max_triples:
        individual = random.choice(individual_names)
        selected_class_set = random.choice(all_disjoint_classes)
        selected_classes = random.sample(selected_class_set, 2)
        triples = {(individual, RDF.type, URIRef(uri + cls)) for cls in selected_classes}
        for triple in triples:
            noisy_g_disjoint.add(triple)
            num_triples +=1
    return noisy_g_disjoint

def add_noise_disjoint_properties(g, g_no_noise, max_triples, all_disjoint_properties, uri):
    noisy_g_disjoint = rdflib.Graph()
    _, individual_names, _ = get_individuals(g_no_noise)
    max_combinations = len(individual_names) * sum(len(class_set) * (len(class_set) - 1) // 2 for class_set in all_disjoint_properties)
    if max_triples > max_combinations:
        needed_individuals = (max_triples - max_combinations) // (max_combinations // len(individual_names)) + 1
        individual_names.extend([URIRef(uri + f"I_{i}") for i in range(1, needed_individuals + 1)])
        print('We created new individuals...')

    num_triples = 0
    while num_triples < max_triples:
        individual = random.choice(individual_names)
        selected_property_set = random.choice(all_disjoint_properties)
        selected_properties = random.sample(selected_property_set, 2)
        _, objects = get_subjects_objects_given_predicate(g, selected_properties, uri)
        selected_object = random.choice(objects)
        triples = {(individual, URIRef(uri + p), URIRef(uri + selected_object)) for p in selected_properties}
        for triple in triples:
            noisy_g_disjoint.add(triple)
            num_triples +=1
    return noisy_g_disjoint

def get_possible_predicates(g_no_noise):
    relations = list(set(g_no_noise.predicates()))

    literal_predicates = set()
    for _, pred, obj in g_no_noise:
        if isinstance(obj, Literal):
            literal_predicates.add(pred)

    possible_predicates = [str(rel) for rel in relations if not rel.startswith('http://www.w3.org/') and not rel in literal_predicates]
    return possible_predicates