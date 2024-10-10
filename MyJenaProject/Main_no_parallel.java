// javac -cp ".;lib/*" Main_no_parallel.java
// java -cp ".;lib/*" Main_no_parallel

import org.apache.jena.rdf.model.*;
import org.apache.jena.reasoner.Reasoner;
import org.apache.jena.reasoner.ReasonerRegistry;
import org.apache.jena.util.FileManager;
import org.apache.jena.rdf.model.InfModel;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class Main_no_parallel {
    public static void main(String[] args) {
        // Path to the directory containing your ontologies
        File dir = new File("input/"); // Change this to your ontologies folder
        File[] files = dir.listFiles((d, name) -> name.endsWith(".owl"));

        // Check if files were found
        if (files != null) {
            for (File ontologyFile : files) {
                processOntology(ontologyFile);
            }
        } else {
            System.out.println("No OWL files found in the specified directory.");
        }
    }

    private static void processOntology(File inputFile) {
        try {
            // Load the ontology file
            Model model = ModelFactory.createDefaultModel();
            FileManager.get().readModel(model, inputFile.getAbsolutePath());

            // Create a reasoner
            Reasoner reasoner = ReasonerRegistry.getOWLReasoner();

            // Bind the model to the reasoner
            InfModel infModel = ModelFactory.createInfModel(reasoner, model);

            // Create a new OWL file to store inferences
            String outputFileName = "output/" + inputFile.getName().replace(".owl", "_inferred.owl");
            try (FileOutputStream out = new FileOutputStream(outputFileName)) {
                infModel.write(out, "RDF/XML");
                System.out.println("Inferences written to: " + outputFileName);
            }
        } catch (IOException e) {
            System.err.println("Error processing " + inputFile.getName() + ": " + e.getMessage());
        }
    }
}
