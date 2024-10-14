// javac -cp ".;lib/*" Main.java
// java -cp ".;lib/*" Main

// javac -cp ".:lib/*" Main.java
// java -cp ".:lib/*" Main

import org.apache.jena.rdf.model.*;
import org.apache.jena.reasoner.Reasoner;
import org.apache.jena.reasoner.ReasonerRegistry;
import org.apache.jena.util.FileManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Main {
    public static void main(String[] args) {
        // Path to the directory containing your ontologies
        File dir = new File("input/"); // Change this to your ontologies folder
        File[] files = dir.listFiles((d, name) -> name.endsWith(".ttl")); // Change .owl to .ttl

        // Check if files were found
        if (files != null && files.length > 0) {
            // Create a fixed thread pool based on available processors
            int numThreads = Runtime.getRuntime().availableProcessors();
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);

            for (File ontologyFile : files) {
                // Submit each ontology processing task to the executor
                executor.submit(() -> processOntology(ontologyFile));
            }

            // Shutdown the executor and wait for all tasks to finish
            executor.shutdown();
            try {
                if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                    System.err.println("Timed out waiting for tasks to finish.");
                }
            } catch (InterruptedException e) {
                System.err.println("Thread execution interrupted: " + e.getMessage());
            }
        } else {
            System.out.println("No TTL files found in the specified directory.");
        }
    }

    private static void processOntology(File inputFile) {
        try {
            // Load the ontology file
            Model model = ModelFactory.createDefaultModel();
            FileManager.get().readModel(model, inputFile.getAbsolutePath());
    
            // Create a reasoner
            // Reasoner reasoner = ReasonerRegistry.getRDFSReasoner();
            Reasoner reasoner = ReasonerRegistry.getOWLReasoner();
    
            // Bind the model to the reasoner
            InfModel infModel = ModelFactory.createInfModel(reasoner, model);
    
            // Create a new TTL file to store inferences
            String outputFileName = "output/" + inputFile.getName().replace(".ttl", "_inferred.ttl"); 
            try (FileOutputStream out = new FileOutputStream(outputFileName)) {
                // Write the header for Turtle format
                out.write("@prefix ns1: <http://benchmark/OWL2Bench#> .\n".getBytes());
                out.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n".getBytes());
                out.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n".getBytes());
                out.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n".getBytes());
                out.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n".getBytes());
    
                // Write all statements, excluding blank nodes
                StmtIterator iter = infModel.listStatements();
                while (iter.hasNext()) {
                    Statement stmt = iter.nextStatement();
                    Resource subject = stmt.getSubject();
                    Property predicate = stmt.getPredicate();
                    RDFNode object = stmt.getObject();
    
                    // Skip blank nodes
                    if (subject.isAnon() || object.isAnon()) {
                        continue; // Skip statements with blank nodes
                    }
    
                    // Prepare the output line
                    StringBuilder outputLine = new StringBuilder();
                    outputLine.append("<").append(subject.toString()).append("> ")
                              .append("<").append(predicate.toString()).append("> ");
    
                    // Check if the object is a URI resource or a literal
                    if (object.isURIResource()) {
                        outputLine.append("<").append(object.toString()).append(">");
                    } else if (object.isLiteral()) {
                        outputLine.append("\"").append(object.asLiteral().getString()).append("\"");
                        if (object.asLiteral().getLanguage() != null) {
                            outputLine.append("^^<http://www.w3.org/2001/XMLSchema#string>"); // Optional datatype
                        }
                    } else {
                        continue; // Skip if the object is not a URI or literal
                    }
                    outputLine.append(" .\n");
    
                    // Write the output line to the file
                    out.write(outputLine.toString().getBytes());
                }
    
                System.out.println("Inferences written to: " + outputFileName);
            }
        } catch (IOException e) {
            System.err.println("Error processing " + inputFile.getName() + ": " + e.getMessage());
        }
    }    
}
