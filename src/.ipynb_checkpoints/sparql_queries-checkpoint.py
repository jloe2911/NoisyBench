import rdflib
from rdflib import URIRef
from src.utils import *

def get_queries():
    
    query1 = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     #?personA rdf:type owl:NamedIndividual .
     #?personB rdf:type owl:NamedIndividual .
     
     { ?personA fo:hasFather ?personB .
       ?personB fo:hasFather ?personA . }
       
     UNION
     
    { ?personA fo:hasMother ?personB .
      ?personB fo:hasMother ?personA . }
     
     FILTER (?personA != ?personB)
    }
    """

    query2 = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     #?personA rdf:type owl:NamedIndividual .
     #?personB rdf:type owl:NamedIndividual .
     
     { ?personA fo:hasSon ?personB .
       ?personB fo:hasSon ?personA . }
       
     UNION
     
    { ?personA fo:hasDaughter ?personB .
      ?personB fo:hasDaughter ?personA . }
     
     FILTER (?personA != ?personB)
    }
    """
    
    query3 = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     #?personA rdf:type owl:NamedIndividual .
     #?personB rdf:type owl:NamedIndividual .

     { ?personA fo:hasSister ?personB }
     UNION
     { ?personA fo:hasBrother ?personB } .
     
     { ?personA fo:hasDaughter ?personB }
     UNION
     { ?personA fo:hasSon ?personB } .
     
     { ?personA fo:hasMother ?personB }
     UNION
     { ?personA fo:hasFather ?personB } .

     FILTER (?personA != ?personB)
    }
    """
    
    return query1, query2, query3

def add_links(g, node1_lst, node2_lst, edge_type_uri):
  
    for node1, node2 in zip(node1_lst,node2_lst):
        g.add((node1,edge_type_uri,node2))

    return g

def print_result(g, query):
    qres = g.query(query)
    for row in qres:
        print(f"{row.contradictions}")