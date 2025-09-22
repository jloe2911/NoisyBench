from sys import argv
import owlready2
import owlready2.reasoning

JAVA_EXE, JAVA_MEM, BASE_URI, OUTPUT = argv[1:]

owlready2.JAVA_EXE = JAVA_EXE
owlready2.reasoning.JAVA_MEMORY = JAVA_MEM

owlready2.onto_path.append("temp")

onto = owlready2.get_ontology(BASE_URI).load()

with onto:
    owlready2.sync_reasoner_pellet(infer_property_values=True)

# Save the inferred ontology
onto.save(file=OUTPUT)