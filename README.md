## Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Tour

* NoiseGeneratation.ipynb: 
Generates ABox noise in an ontology: a method based on disjoint axioms, a Graph Neural Network (GNN) approach and a random-based approach. For each noise generation method, we introduced a parameter $k$, where $k$ indicates the percentage of noise to be added in relation to the total number of triples of the original ontology.
* GenerateGraphs.ipynb: 
For each resource $r$, a small graph $g$ is created using the following command: \colorbox{lightgray}{\texttt{DESCRIBE <r>}}. This command retrieves all statements related to the resource $r$ for inference.
* MyJenaProject: 
Applies Jena to graphs $g_1$, $g_2$, ..., $g_R$, where $R$ represents the set of resources from the original ontology. This generates inference graphs $i_1$, $i_2$, ..., $i_R$. The idea is that each subgraph $g$ and its corresponding inference graph $i$ form a feature-label pair $(g, i)$.
* PrepareGraphs.ipynb: 
To construct the training, test and validation sets, each $(g, i)$ pair is assigned to one of these sets using a stratified splitting technique. Specifically, the training set $G_{\text{train}}$ will consist of a collection of $(g, i)$ graphs, with a similar process followed for the test and validation sets.
* OWL2Vec.ipynb: OWL2Vec*.
* EL.ipynb: Box2EL.