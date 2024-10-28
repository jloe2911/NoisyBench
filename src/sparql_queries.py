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
     
     { ?personA fo:isFatherOf ?personB .
       ?personB fo:isFatherOf ?personA . }
       
     UNION
     
    { ?personA fo:isMotherOf ?personB .
      ?personB fo:isMotherOf ?personA . }
     
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
     
     { ?personA fo:isSonOf ?personB .
       ?personB fo:isSonOf ?personA . }
       
     UNION
     
    { ?personA fo:isDaughterOf ?personB .
      ?personB fo:isDaughterOf ?personA . }
     
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

     { ?personA fo:isSisterOf ?personB .
       ?personA fo:isDaughterOf ?personB .
       ?personA fo:isMotherOf ?personB . }

     UNION
     
     { ?personA fo:isBrotherOf ?personB .
       ?personA fo:isSonOf ?personB .
       ?personA fo:isFatherOf ?personB . }

     FILTER (?personA != ?personB)
    }
    """
    
    query4 = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     #?personA rdf:type owl:NamedIndividual .
     #?personB rdf:type owl:NamedIndividual .

     { ?personA fo:isFemalePartnerIn ?personB .
       ?personB fo:isFemalePartnerIn ?personC . }

     UNION

     { ?personA fo:isMalePartnerIn ?personB .
       ?personB fo:isMalePartnerIn ?personC . }

     FILTER (?personA != ?personB && ?personB != ?personC) 
    }
    """
    
    query5 = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     #?personA rdf:type owl:NamedIndividual .
     #?personB rdf:type owl:NamedIndividual .

     { ?personA fo:isMotherOf ?personB ;
                fo:hasBirthYear ?personABirthDate . 
       ?personB fo:hasBirthYear ?personBBirthDate . }        

     UNION

     { ?personA fo:isFatherOf ?personB ;
                fo:hasBirthYear ?personABirthDate . 
       ?personB fo:hasBirthYear ?personBBirthDate . }        

     FILTER (?personA != ?personB && ?personABirthDate < ?personBBirthDate) 
    }
    """
    
    query6 = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     #?personA rdf:type owl:NamedIndividual .
     #?personB rdf:type owl:NamedIndividual .

     { ?personA fo:isDaughterOf ?personB ;
                fo:hasBirthYear ?personABirthDate . 
       ?personB fo:hasBirthYear ?personBBirthDate . }        

     UNION

     { ?personA fo:isSonOf ?personB ;
                fo:hasBirthYear ?personABirthDate . 
       ?personB fo:hasBirthYear ?personBBirthDate . }        

     FILTER (?personA != ?personB && ?personABirthDate > ?personBBirthDate) 
    }
    """
    
    query7 = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     #?personA rdf:type owl:NamedIndividual .
     #?personB rdf:type owl:NamedIndividual .
     ?personA ?p1 ?personB .
     ?personA ?p2 ?personB .
     FILTER (?p1 in (fo:isSisterOf, fo:isDaughterOf, fo:isMotherOf) && ?p2 in (fo:isFatherOf, fo:isBrotherOf, fo:isSonOf))
     FILTER (?personA != ?personB)
    }
    """
    
    return query1, query2, query3, query4, query5, query6, query7

def add_links(g, node1_lst, node2_lst, edge_type_uri):
  
    for node1, node2 in zip(node1_lst,node2_lst):
        g.add((node1,edge_type_uri,node2))

    return g

def print_result(g, query):
    qres = g.query(query)
    for row in qres:
        print(f"{row.contradictions}")