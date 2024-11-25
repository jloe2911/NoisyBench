import rdflib
from rdflib import Literal, RDF, OWL
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
        print('not enough disjoint classes')
    #     needed_individuals = (max_triples - max_combinations) // (max_combinations // len(individual_names)) + 1
    #     individual_names.extend([URIRef(uri + f"I_{i}") for i in range(1, needed_individuals + 1)])
    #     print('We created new individuals...')

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
        print('not enough disjoint classes')
    #     needed_individuals = (max_triples - max_combinations) // (max_combinations // len(individual_names)) + 1
    #     individual_names.extend([URIRef(uri + f"I_{i}") for i in range(1, needed_individuals + 1)])
    #     print('We created new individuals...')

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

    individual_predicates = set()
    
    for subj, pred, obj in g_no_noise:
        if (subj, RDF.type, OWL.NamedIndividual) in g_no_noise and \
           (obj, RDF.type, OWL.NamedIndividual) in g_no_noise and \
           not isinstance(obj, Literal):
            individual_predicates.add(pred)
    
    possible_predicates = [str(pred) for pred in individual_predicates if not str(pred).startswith('http://www.w3.org/')]
    
    return possible_predicates

def get_non_domain_individuals(g_no_noise, domain_range):
    non_domain_individuals = []
    for subj in g_no_noise.subjects(RDF.type, OWL.NamedIndividual):
        for _, _, obj in g_no_noise.triples((subj, RDF.type, None)):
            if obj != OWL.NamedIndividual and obj != domain_range:
                non_domain_individuals.append(subj)
    return non_domain_individuals

def violate_domain(g_no_noise, range_domain_info, non_range_domain_individuals_dict, k):    
    violated_g = rdflib.Graph()
    properties = list(range_domain_info.keys())  
    count = 0
    
    property_triples = {}
    for prop, info in range_domain_info.items():
        if "domain" in info:
            domain = URIRef(info["domain"])
            predicate = URIRef(prop)
            existing_triples = list(g_no_noise.triples((None, predicate, None)))
            property_triples[prop] = (domain, existing_triples)
    
    if not property_triples:
        return violated_g

    while count < k:
        prop = random.choice(properties)
        domain, existing_triples = property_triples.get(prop, (None, []))
        
        if not existing_triples:
            continue
        
        non_domain_individuals = non_range_domain_individuals_dict[domain]
        
        if not non_domain_individuals:
            continue

        _, pred, obj = random.choice(existing_triples)
        violating_subject = random.choice(non_domain_individuals)
        violated_g.add((violating_subject, pred, obj))
        count += 1

    return violated_g

def violate_range(g_no_noise, range_domain_info, non_range_domain_individuals_dict, k):    
    violated_g = rdflib.Graph()
    properties = list(range_domain_info.keys())  
    count = 0

    property_triples = {}
    for prop, info in range_domain_info.items():
        if "range" in info:
            range_ = URIRef(info["range"])
            predicate = URIRef(prop)
            existing_triples = list(g_no_noise.triples((None, predicate, None)))
            property_triples[prop] = (range_, existing_triples)
    
    if not property_triples:
        return violated_g

    while count < k:
        prop = random.choice(properties)
        range_, existing_triples = property_triples.get(prop, (None, []))
        
        if not existing_triples:
            continue

        non_range_individuals = non_range_domain_individuals_dict[range_]
        
        if not non_range_individuals:
            continue

        subj, pred, _ = random.choice(existing_triples)
        violating_object = random.choice(non_range_individuals)
        violated_g.add((subj, pred, violating_object))
        count += 1

    return violated_g