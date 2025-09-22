# NSORN: Designing a Benchmark Dataset for Neurosymbolic Ontology Reasoning with Noise

**Benchmarking framework for evaluating neurosymbolic ontology reasoners under noisy conditions.**

**NSORN** introduces three types of noise—**logical, random, and statistical**—into the ABox of ontologies to test the robustness of neurosymbolic reasoners. We provide noisy benchmarks for widely used ontologies (e.g., **OWL2Bench** and **Family**) and evaluate state-of-the-art neurosymbolic reasoners (e.g., **Box2EL** and **OWL2Vec***). Our findings show that logical noise is the most challenging, causing significant drops in reasoning performance.

---

## Features

- 🧩 **Noise Injection Framework** – Add logical, random, and statistical noise to ontologies.  
- 📊 **Benchmark Creation** – Generate noisy versions of widely used ontologies.  
- ⚡ **Reasoner Evaluation** – Assess neurosymbolic reasoners under different noise conditions.  
- 🔍 **Reasoning Tasks Support** – Evaluate tasks like class membership and object property assertions.  
- 📉 **Performance Insights** – Study how different types of noise impact reasoning robustness.

---

## Requirements

- Python 3.12.7  
- pip (Python package manager)

---

## Installation

**Clone the repository:**

```bash
git clone https://github.com/jloe2911/NoisyBench
cd noisybench
```

**(Option 1): Install dependencies within virtual environment**

```bash
# Create a virtual environment named 'env'

python3 -m venv env

# Activate the environment

# On macOS/Linux

source env/bin/activate

# On Windows

env\Scripts\activate

# Install dependencies inside the environment

pip install -r requirements.txt
```

**(Option 2): Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Usage

### Create Noisy Ontologies

```bash
python pipeline_noise.py --dataset_name family
python pipeline_noise.py --dataset_name OWL2DL-1
```

The pipeline uses ontologies stored in the `ontology` folder:

- `family.owl`
- `OWL2Bench.owl`

For each ontology, two additional versions are provided:

- `*_TBOX.owl`: contains only the TBox.
- `*_modified.owl`: enriched with additional disjoint classes and properties.

**Pipeline Steps:**

1. **Graph Construction per Resource**  
   - For each resource `r`, construct a small graph `g` containing all triples where `r` appears as subject or object.
   - Each graph `g`g is expanded to two hops (`g'`) to include all statements within two hops of `r`.
   - The TBox is added to each graph.

   **Outputs:**  
   - **Input graphs:** 2-hop + TBox  
   - **Filtered 1-hop graphs:** 1-hop without TBox  
   - **Filtered input graphs:** 2-hop without TBox  

2. **Run the Reasoner**  
   - Apply Pellet to the **input graphs** to produce **inferred_graphs**.

3. **Filter Inferences**
   - To extract only meaningful triples, we focus on membership and property assertion triples, removing any triples where the object is a `Literal` or `owl:Thing`, yielding refined graphs **inferred_graphs_filtered**.

4. **Build Train, Test and Validation Graphs**
   - Since our approach is unsupervised, the ontology is added to `G_{train}`, while graphs from **inferred_graphs_filtered** are assigned to `G_{test}` and `$G_{val}`.

5. **Create Noise**
   - Generates ABox noise in an ontology: random, statistical and logical contradictions.

   **Outputs:**  
| Noise Level | Description |
|------------|-------------|
| 0.25       | 25% of the original triples are modified/added as noise |
| 0.5        | 50% of the original triples are modified/added as noise |
| 0.75       | 75% of the original triples are modified/added as noise |
| 1.0        | 100% of the original triples are modified/added as noise |

### To Run NeuroSymbolic Reasoners

```bash
python pipeline_reasoner.py --dataset_name family --reasoner owl2vec
```