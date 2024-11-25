# Introducing Noise: Designing a Benchmark Dataset for Neuro-Symbolic Ontology Reasoners

## Description

In the field of neuro-symbolic computing, there is a significant gap in standardized benchmark datasets specifically designed for evaluating neuro-symbolic reasoning systems. Currently, no benchmarks or evaluation frameworks have been explicitly developed for this purpose. Existing datasets largely focus on assessing symbolic reasoning with static data and are not tailored to unique challenges of integrating neuro-symbolic methods. Consequently, most evaluations rely on publicly available ontologies, which fail to address the distinct needs of neuro-symbolic integration. Therefore, this paper aims to develop a mechanism for introducing noise into ontologies, particularly focusing on the ABox, and evaluates the performance of existing neuro-symbolic reasoners under varying noise levels. We developed three techniques to introduce noise into ontologies, focusing on the ABox: logical contradictions, statistical contradictions, and random contradictions. Logical contradictions involve violations of disjoint axioms and domain/range constraints, while statistical contradictions use Graph Neural Networks (GNNs) to add low-probability links as noise. These methods were designed to assess the performance and robustness of ontology reasoning under different types of noise. We introduced noise to OWL2Bench and tested the resulting benchmarks on state-of-the-art neuro-symbolic reasoners, Box2EL and OWL2Vec*. To evaluate how these reasoners handle noise, we focused on specific reasoning tasks such as class and object property assertions. Our main finding is that disjoint axioms create a more challenging learning environment, resulting in decreased performance for both Box2EL and OWL2Vec*.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Tour

1. `GenerateGraphs.ipynb`: 
For each resource $r$, we construct a small graph $g$ that includes all triples where either the subject or object is $r$. We divide the original ontology into these smaller graphs $g$ to improve Pellet's scalability. To ensure effective inference, each graph $g$ is extended to two hops, denoted $g'$, capturing all statements within two hops of $r$, and the TBox is added to each graph $g$.
2. `Owlready2-example`: 
We then apply Pellet to the extended graphs $g'_1$, $g'_2$, ..., $g'_R$, where $R$ represents the set of resources in the original ontology, resulting in the inference graphs $i_1$, $i_2$, ..., $i_R$. 
3. `PrepareGraphs.ipynb`: 
To extract only meaningful triples, we focus on membership, subsumption, and property assertion triples, removing any triples where the object is a Literal or owl:Thing, yielding refined graphs $i^{*}_1$, $i^{*}_2$, ..., $i^{*}R$. Since our approach is unsupervised, the graphs $g_1$, $g_2$, ..., $g_R$ are ultimately added to $G_{train}$, while $i^{*}1$, $i^{*}_2$, ..., $i^{*}R$ are assigned to $G_{test}$ and $G_{val}$ using a stratified splitting technique.
4. `NoiseGeneratation.ipynb`: 
Generates ABox noise in an ontology: logical contradictions, statistical contradictions, and random contradictions. For each noise generation method, we introduced a parameter $k$, where $k$ indicates the percentage of noise to be added in relation to the total number of triples of the original ontology. We use OWL2DL-1 ontology that can be found [here](https://drive.google.com/drive/folders/1HYURRLaQkLK8cQwV-UBNKK4_Zur2nU68?usp=drive_link) and should be added to `datasets/{dataset_name}.owl`. The notebook outputs noisy datasets stored in `datasets/noise/{dataset_name}_{noise_generation_method}_{noise_percentage}.owl`. The noisy datasets can also be found [here](https://drive.google.com/drive/folders/14TzofCSdxgvXEA5aJ7fhppK8EuN5k2aH?usp=drive_link).
5. `OWL2Vec.ipynb`: OWL2Vec*.
6. `EL.ipynb`: Box2EL.