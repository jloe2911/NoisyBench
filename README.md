# NSORN: Designing a Benchmark Dataset for Neurosymbolic Ontology Reasoning with Noise

**Benchmarking framework for evaluating neurosymbolic ontology reasoners under noisy conditions.**

**NSORN** introduces three types of noise—**logical, random, and statistical**—into the ABox of ontologies to test the robustness of neurosymbolic reasoners. We provide noisy benchmarks for widely used ontologies (e.g., **OWL2DL-1**, **Family** and **Pizza**) and evaluate state-of-the-art neurosymbolic reasoners (e.g., **Box2EL**, **OWL2Vec** and **RGCN**). Our findings show that logical noise is the most challenging, causing significant drops in reasoning performance.

Using the **Pizza ontology**, we created an ABox generator to support experiments with synthetic data (`see ontologies/Abox-generation.ipynb`). The process for generating ABox data for the Pizza ontology begins by loading the Pizza TBox (Terminological Box) axioms. A custom instance generation step then automatically creates a specified number of individuals (ABox data), and their object properties based on a configuration. For this study, we use only `NamedPizza` class and `hasTopping` property in the configuration. Crucially, this generation leverages the TBox's inherent OWL restrictions (e.g.,`only` or `some` constraints) to dynamically determine the appropriate target classes for object properties, thereby guaranteeing the generated ABox is semantically consistent with the ontology's definition. The final output is the complete ontology, comprising the original TBox and the newly populated ABox.

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
- Java Version 22.0.2 (Released: 2024-07-16)

---

## Installation

**Clone the repository:**

```bash
git clone https://github.com/jloe2911/NoisyBench
cd noisybench
```

**Install dependencies within virtual environment**

```bash
# Create a virtual environment named 'env'

python -m venv env

# Activate the environment

# On macOS/Linux

source env/bin/activate

# On Windows

env\Scripts\activate

# Install dependencies inside the environment

pip install -r requirements.txt
```

**Setting up Java**

We use the `mowl` library to run the reasoners, which depend on Java. However, Java is not required if your goal is only to create noisy ontologies. To make sure everything runs smoothly, some additional configuration is needed:

1. Java Installation

- Make sure you have **Java JDK** installed on your system.
- Find the installation path. Examples:
  - Windows: `C:\Program Files\Java\jdk-22`
  - Mac/Linux: `/usr/lib/jvm/java-22-openjdk`

2. Set `JAVA_HOME`

- The notebook relies on the `JAVA_HOME` environment variable.
- In `consts.py`:

```bash
import os

JAVA_HOME_PATH = r"C:\Program Files\Java\jdk-25"  # <-- Change this if needed
os.environ["JAVA_HOME"] = JAVA_HOME_PATH
```

3. Initialize JVM Memory

- `mowl` runs on the Java Virtual Machine (JVM). You need to allocate memory:

```bash
import mowl
from consts import JVM_MEMORY

mowl.init_jvm(JVM_MEMORY)
```

- Default: '10g' (10 GB)
- Warning: If your computer has less RAM, reduce this value, e.g., '4g'.

---

## Usage

### Create Noisy Ontologies

```bash
python split.py --dataset_name family --tbox family_TBOX
python pipeline_noise.py --dataset_name family --steps noise

python split.py --dataset_name pizza_100 --tbox pizza_TBOX
python pipeline_noise.py --dataset_name pizza_100 --steps noise

python split.py --dataset_name pizza_250 --tbox pizza_TBOX
python pipeline_noise.py --dataset_name pizza_250 --steps noise

python split.py --dataset_name OWL2DL-1 --tbox OWL2DL-1_TBOX
python pipeline_noise.py --dataset_name OWL2DL-1 --steps noise
```

The pipeline uses ontologies stored in the `ontology` folder:

- `family.owl`
- `pizza_100.owl`
- `pizza_250.owl`
- `OWL2DL-1.owl`

For each ontology, two additional versions are provided:

- `*_TBOX.owl`: contains only the TBox.
- `*_inferred.owl`: contains the inferred axioms using Pellet (Protege).

**Pipeline Steps:**

1. **Build Train, Test and Validation Graphs**
   - Since our approach is unsupervised, the ontology is added to `G_{train}`, while the inferences are randomly assigned to `G_{train}`, `G_{test}` and `G_{val}`.

2. **Create Noise**
   - Generates ABox noise in an ontology: random, statistical and logical contradictions.

   **Outputs:**  
   - **0.25:** 25% of the original triples are modified/added as noise.
   - **0.50:** 50% of the original triples are modified/added as noise.
   - **0.75:** 75% of the original triples are modified/added as noise.
   - **1.00:** 100% of the original triples are modified/added as noise.

### To Run NeuroSymbolic Reasoners

```bash
python pipeline_reasoner.py --dataset_name family --reasoner owl2vec
python pipeline_reasoner.py --dataset_name family --reasoner box2el
python pipeline_reasoner.py --dataset_name family --reasoner rgcn

python pipeline_reasoner.py --dataset_name pizza_100 --reasoner owl2vec
python pipeline_reasoner.py --dataset_name pizza_100 --reasoner box2el
python pipeline_reasoner.py --dataset_name pizza_100 --reasoner rgcn

python pipeline_reasoner.py --dataset_name pizza_250 --reasoner owl2vec
python pipeline_reasoner.py --dataset_name pizza_250 --reasoner box2el
python pipeline_reasoner.py --dataset_name pizza_250 --reasoner rgcn

python pipeline_reasoner.py --dataset_name OWL2DL-1 --reasoner owl2vec
python pipeline_reasoner.py --dataset_name OWL2DL-1 --reasoner box2el
python pipeline_reasoner.py --dataset_name OWL2DL-1 --reasoner rgcn
```