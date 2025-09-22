import numpy as np
from datetime import datetime
import rdflib
from rdflib import URIRef

import mowl
mowl.init_jvm('10g')
from mowl.owlapi import OWLAPIAdapter
from org.semanticweb.owlapi.model.parameters import Imports
from java.util import HashSet

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def copy_graph(g):
    new_g = rdflib.Graph()

    for triple in g:
        new_g.add(triple)
    
    return new_g

def get_G(g, resources):
    describe_graph = rdflib.Graph()
    for subject_resource in resources:
        for triple in g.triples((None, None, URIRef(subject_resource))):
            describe_graph.add(triple) 
        for triple in g.triples((URIRef(subject_resource), None, None)):
            describe_graph.add(triple) 
    return describe_graph

def create_G(triples):
    G = rdflib.Graph()
    for triple in triples:
        G.add(triple)
    return G

def get_individuals(g):    

    qres = g.query("""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?s WHERE {
    ?s rdf:type owl:NamedIndividual .
    }
    """)

    individuals = []
    for row in qres:
        individuals.append(row.s)

    nodes = list(set(g.subjects()).union(set(g.objects())))
    nodes_dict = {node: i for i, node in enumerate(nodes)}
    nodes_dict_rev = {value: key for key, value in nodes_dict.items()}

    individual_id = [nodes_dict[individual] for individual in individuals]
    individual_names = [nodes_dict_rev[individual_id] for individual_id in individual_id]
    individual_names_dict = dict(zip(list(set(np.arange(len(individual_id)))), individual_names))

    return individual_id, individual_names, individual_names_dict

def get_subjects_objects_given_predicate(g, predicates, uri):
    subjects = []
    objects = []
    for predicate in predicates:
        subjects.extend([subj for subj, _, _ in g.triples((None, URIRef(uri + predicate), None))])
        objects.extend([obj for _, _, obj in g.triples((None, URIRef(uri + predicate), None))])
    return list(set(subjects)), list(set(objects)) 

def add_links(g, node1_lst, node2_lst, edge_type_uri):
    for node1, node2 in zip(node1_lst,node2_lst):
        g.add((node1, edge_type_uri, node2))
    return g

def preprocess_ontology_el(ontology):
    tbox_axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))
    abox_axioms = ontology.getABoxAxioms(Imports.fromBoolean(True))
    new_tbox_axioms = HashSet() 
    for axiom in tbox_axioms:
        axiom_as_str = axiom.toString()
        if "ObjectHasValue" in axiom_as_str:
            continue
        elif "DataSomeValuesFrom" in axiom_as_str:
            continue
        elif "DataAllValuesFrom" in axiom_as_str:
            continue
        elif "DataHasValue" in axiom_as_str:
            continue
        elif "DataPropertyRange" in axiom_as_str:
            continue
        elif "DataPropertyDomain" in axiom_as_str:
            continue
        elif "FunctionalDataProperty" in axiom_as_str:
            continue
        elif "DisjointUnion" in axiom_as_str:
            continue
        elif "HasKey" in axiom_as_str:
            continue
        new_tbox_axioms.add(axiom)
    owl_manager = OWLAPIAdapter().owl_manager
    new_ontology = owl_manager.createOntology(new_tbox_axioms)
    new_ontology.addAxioms(abox_axioms)
    return new_ontology

def get_experiments(dataset_name):
    
    if dataset_name == 'family': 
    
        experiments = [{'dataset_name' : 'family',
                        'file_name' : 'family',
                        'format_' : None,
                        'add_noise': False},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_gnn_0.25',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_random_0.25',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_logical_0.25',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_gnn_0.5',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_random_0.5',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_logical_0.5',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_gnn_0.75',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_random_0.75',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_logical_0.75',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_gnn_1.0',
                        'format_' : None,
                        'add_noise': True},                 
                       {'dataset_name' : 'family',
                        'file_name' : 'family_random_1.0',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'family',
                        'file_name' : 'family_logical_1.0',
                        'format_' : None,
                        'add_noise': True}]
        
    elif dataset_name == 'OWL2DL-1':
    
        experiments = [{'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1',
                        'format_' : None,
                        'add_noise': False},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_gnn_0.25',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_random_0.25',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_logical_0.25',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_gnn_0.5',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_random_0.5',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_logical_0.5',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_gnn_0.75',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_random_0.75',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_logical_0.75',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_gnn_1.0',
                        'format_' : None,
                        'add_noise': True},                 
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_random_1.0',
                        'format_' : None,
                        'add_noise': True},
                       {'dataset_name' : 'OWL2DL-1',
                        'file_name' : 'OWL2DL-1_logical_1.0',
                        'format_' : None,
                        'add_noise': True}]

    return experiments

def save_results(subsumption_results, membership_results, link_prediction_results, results_dir):
    with open(results_dir, 'w') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Results as of {timestamp}:\n\n")
        
        # Function to format metrics
        def format_metrics(name, results):
            results_arr = np.array(results)
            lines = [f"{name}:\n"]
            for i, row in enumerate(results_arr):
                lines.append(f"Run {i+1}: " + " & ".join([f"{x:.3f}" for x in row]) + "\n")
            means = np.mean(results_arr, axis=0)
            lines.append("Mean: " + " & ".join([f"{x:.3f}" for x in means]) + "\n\n")
            return "".join(lines)
        
        # Write all results
        f.write(format_metrics("Membership", membership_results))
        f.write(format_metrics("Subsumption", subsumption_results))
        f.write(format_metrics("Link Prediction", link_prediction_results))
    
    logger.info(f"Results saved to {results_dir}")