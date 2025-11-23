import argparse
import os
from consts import JAVA_HOME_PATH
os.environ["JAVA_HOME"] = JAVA_HOME_PATH

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.utils import get_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name", choices=['pizza', 'pizza_100', 'pizza_250', 'family', 'OWL2DL-1_100'])
    parser.add_argument("--reasoner", type=str, required=True, help="Neurosymbolic reasoner to use", choices=["owl2vec", "box2el", "rgcn"])
    args = parser.parse_args()

    dataset_name = args.dataset_name
    reasoner = args.reasoner

    experiments = get_experiments(dataset_name)

    if reasoner == 'owl2vec':
        from src.owl2vec import run_owl2vec
        run_owl2vec(dataset_name, 'cpu', experiments)

    elif reasoner == 'box2el':
        from src.elmodule import run_box2el
        run_box2el('cpu', experiments)

    elif reasoner == 'rgcn':
        from src.gnn import run_rgcn
        run_rgcn('cpu', experiments)
