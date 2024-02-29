import rdflib

def get_queries():
    
    '''Contradictory Parentage: Person A is a parent of Person B however Person B is also listed to be a parent of Person A, 
    creating a loop or contradiction.'''

    query1 = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     ?personA fo:hasFather ?personB .
     ?personB fo:hasFather ?personA .
     FILTER (?personA != ?personB)
    }
    """

    query2 = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fo: <http://www.co-ode.org/roberts/family-tree.owl#>
    SELECT (COUNT(*) as ?contradictions) WHERE {
     ?personA fo:hasMother ?personB .
     ?personB fo:hasMother ?personA .
     FILTER (?personA != ?personB)
    }
    """
    
    return query1, query2

def print_result(g, query):
    qres = g.query(query)
    for row in qres:
        print(f"{row.contradictions}")