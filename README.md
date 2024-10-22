# Introducing Noise: Designing a Benchmark Dataset for Neuro-Symbolic Ontology Reasoners

## Description

In the field of neuro-symbolic computing, there is a significant gap in standardized benchmark datasets specifically designed for evaluating neuro-symbolic reasoning systems. Currently, no benchmarks or evaluation frameworks have been explicitly developed for this purpose. Existing datasets largely focus on assessing symbolic reasoning with static data and are not tailored to unique challenges of integrating neuro-symbolic methods. Consequently, most evaluations rely on publicly available ontologies, which fail to address the distinct needs of neuro-symbolic integration. Therefore, this paper aims to develop a mechanism for introducing noise into ontologies, particularly focusing on the ABox, and evaluates the performance of existing neuro-symbolic reasoners under varying noise levels. We developed three distinct techniques to introduce ABox noise in an ontology: a method based on disjoint axioms, a Graph Neural Network (GNN) approach and a random-based technique. We introduced noise to OWL2Bench and tested the resulting benchmarks on state-of-the-art neuro-symbolic reasoners, Box2EL and OWL2Vec*. To evaluate how these reasoners handle noise, we focused on specific reasoning tasks such as membership and subsumption. Our main finding is that disjoint axioms create a more challenging learning environment, resulting in decreased performance for both Box2EL and OWL2Vec*.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Tour

1. `NoiseGeneratation.ipynb`: 
Generates ABox noise in an ontology: a method based on disjoint axioms, a Graph Neural Network (GNN) approach and a random-based approach. For each noise generation method, we introduced a parameter $k$, where $k$ indicates the percentage of noise to be added in relation to the total number of triples of the original ontology.
We use OWL2DL-1 ontology that can be found [here](https://drive.google.com/drive/folders/1HYURRLaQkLK8cQwV-UBNKK4_Zur2nU68?usp=drive_link) and should be added to `datasets/{dataset_name}.owl`.
The notebook outputs noisy datasets stored in `datasets/noise/{dataset_name}_noisy_{noise_generation_method}_{noise_percentage}.owl`. The noisy datasets can also be found [here](https://drive.google.com/drive/folders/14TzofCSdxgvXEA5aJ7fhppK8EuN5k2aH?usp=drive_link).
2. `GenerateGraphs.ipynb`: 
For each resource $r$, a small graph $g$ is created using the following command: DESCRIBE <r>. This command retrieves all statements related to the resource $r$ for inference.
3. `MyJenaProject`: 
Applies Jena to graphs $g_1$, $g_2$, ..., $g_R$, where $R$ represents the set of resources from the original ontology. This generates inference graphs $i_1$, $i_2$, ..., $i_R$. The idea is that each subgraph $g$ and its corresponding inference graph $i$ form a feature-label pair $(g, i)$.
4. `PrepareGraphs.ipynb`: 
To construct the training, test and validation sets, each $(g, i)$ pair is assigned to one of these sets using a stratified splitting technique. Specifically, the training set $G_{\text{train}}$ will consist of a collection of $(g, i)$ graphs, with a similar process followed for the test and validation sets.
5. `OWL2Vec.ipynb`: OWL2Vec*.
6. `EL.ipynb`: Box2EL.