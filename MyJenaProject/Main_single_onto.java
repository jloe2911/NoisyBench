// javac -cp ".;lib/*" Main_single_onto.java
// java -cp ".;lib/*" Main_single_onto

import org.apache.jena.rdf.model.*;
import org.apache.jena.reasoner.Reasoner;
import org.apache.jena.reasoner.ReasonerRegistry;
import org.apache.jena.util.FileManager;
import org.apache.jena.rdf.model.InfModel;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class Main_single_onto {
    public static void main(String[] args) {
        // Specify the ontology file (you can modify the path)
        File ontologyFile = new File("input_toy_example/U0WC0D2AP3.ttl"); // Change to your file path

        // Check if the file exists
        if (ontologyFile.exists()) {
            processOntology(ontologyFile);
        } else {
            System.out.println("The specified ontology file does not exist.");
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

            // Specify the output directory
            String outputDirectory = "output_toy_example/"; // Change this to your desired output folder
            // Create the output directory if it doesn't exist
            new File(outputDirectory).mkdirs();

            // Create a new OWL file to store inferences
            String outputFileName = outputDirectory + inputFile.getName().replace(".ttl", "_inferred.ttl");
            try (FileOutputStream out = new FileOutputStream(outputFileName)) {
                infModel.write(out, "RDF/XML");
                System.out.println("Inferences written to: " + outputFileName);
            }
        } catch (IOException e) {
            System.err.println("Error processing " + inputFile.getName() + ": " + e.getMessage());
        }
    }
}
