// javac -cp ".;lib/*" HermiTBatchProcessor.java
// java -cp ".;lib/*" HermiTBatchProcessor

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.HermiT.Reasoner;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.util.InferredAxiomGenerator;
import org.semanticweb.owlapi.util.InferredClassAssertionAxiomGenerator;
import org.semanticweb.owlapi.util.InferredSubClassAxiomGenerator;
import org.semanticweb.owlapi.util.InferredPropertyAssertionGenerator;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Set;
import java.util.List;
import java.util.ArrayList;

public class HermiTBatchProcessor {

    public static void main(String[] args) {
        File folder = new File("input_toy_example/"); 
        if (!folder.exists() || !folder.isDirectory()) {
            System.out.println("The specified path does not exist or is not a directory: " + folder.getAbsolutePath());
            return;
        }
        
        File[] listOfFiles = folder.listFiles((dir, name) -> name.endsWith("OWL2DL-1.owl")); 

        if (listOfFiles == null || listOfFiles.length == 0) {
            System.out.println("No files found in the specified directory.");
            return;
        }

        for (File file : listOfFiles) {
            if (file.isFile()) {
                System.out.println("Processing file: " + file.getName());
                processOntology(file);
            }
        }
    }

    private static void processOntology(File file) {
        try {
            // Initialize OWL API manager and load ontology
            OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
            OWLOntology ontology = manager.loadOntologyFromOntologyDocument(file);

            // Set up HermiT reasoner
            OWLReasonerFactory reasonerFactory = new Reasoner.ReasonerFactory();
            OWLReasoner reasoner = reasonerFactory.createReasoner(ontology);

            // Perform reasoning tasks and add inferences to ontology
            if (reasoner.isConsistent()) {
                // Generators for inferred axioms
                List<InferredAxiomGenerator<? extends OWLAxiom>> generators = new ArrayList<>();
                generators.add(new InferredSubClassAxiomGenerator());
                generators.add(new InferredClassAssertionAxiomGenerator());
                generators.add(new InferredPropertyAssertionGenerator());

                // Apply each generator to add inferred axioms to ontology
                for (InferredAxiomGenerator<? extends OWLAxiom> generator : generators) {
                    Set<? extends OWLAxiom> inferredAxioms = generator.createAxioms(manager, reasoner);
                    inferredAxioms.forEach(axiom -> manager.addAxiom(ontology, axiom));
                }

                // Save inferred ontology to file
                FileOutputStream out = new FileOutputStream("output_toy_example/" + file.getName().replace(".owl", "_inferred.owl"));
                manager.saveOntology(ontology, out);
                out.close();
            } else {
                System.out.println("Ontology is inconsistent: " + file.getName());
            }

            // Clean up
            reasoner.dispose();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
