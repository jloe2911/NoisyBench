from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import rdflib
from rdflib import URIRef, RDFS, RDF, OWL, Literal, BNode
from owlready2 import get_ontology
import glob

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
    g_inferred_filtered.serialize(filtered_inferred_path, format="turtle")
    return f"Filtered: {filtered_inferred_path}"

def process_filtered_graph(input_graph_path, dataset_name, object_properties):
    try:
        if 'xml' in input_graph_path:
            file = input_graph_path.replace(f'datasets/{dataset_name}_inferred_graphs\\', '').replace('.xml', '')
            input_path = f'datasets/{dataset_name}_input_graphs/{file}.ttl'
            filtered_inferred_path = f'datasets/{dataset_name}_inferred_graphs_filtered/{file}.ttl'
        else:
            file = input_graph_path.replace(f'datasets/{dataset_name}_inferred_graphs\\', '')
            input_path = f'datasets/{dataset_name}_input_graphs/{file}'
            filtered_inferred_path = f'datasets/{dataset_name}_inferred_graphs_filtered/{file}'

        filter_inferred_triples(input_path, input_graph_path, filtered_inferred_path, object_properties)
        return f"Processed: {input_graph_path}"
    except Exception as e:
        return f"Error processing {input_graph_path}: {e}"

if __name__ == "__main__":
    dataset_name = 'OWL2DL-1'  
    dataset_name = 'family'  
    ontology = get_ontology(f'datasets/{dataset_name}.owl').load()
    object_properties = list(ontology.object_properties())
    object_properties = [URIRef(x.iri) for x in object_properties]

    print(f"Loaded {len(object_properties)} object properties.")

    inferred_graph_paths = sorted(glob.glob(f'datasets/{dataset_name}_inferred_graphs/*'))

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_filtered_graph, path, dataset_name, object_properties): path
                   for path in inferred_graph_paths}

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            print(result)
