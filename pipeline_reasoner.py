import argparse

import os
os.environ["JAVA_HOME"] = r"C:\Program Files\JAVA\jdk-22"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.utils import get_experiments
from src.owl2vec import run_owl2vec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name", choices=['family', 'OWL2DL-1'])
    parser.add_argument("--reasoner", type=str, required=True, help="Neurosymbolic reasoner to use", choices=["owl2vec", "box2el"])
    args = parser.parse_args()

    dataset_name = args.dataset_name
    reasoner = args.reasoner

    experiments = get_experiments(dataset_name)

    if reasoner == 'owl2vec':
        run_owl2vec(dataset_name, 'cpu', experiments)
