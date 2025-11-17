import argparse
import subprocess
from construct_graphs import generate_graphs
from filter_inferences import run_filtering
from train_test_val_split import build_rdf_datasets
from add_noise import add_noise_to_dataset

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.utils import get_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True, 
        help="Dataset name", 
        choices=['pizza', 'pizza_100', 'pizza_1000', 'family', 'OWL2DL-1']
    )

    # flags for each step
    parser.add_argument("--steps", nargs="+", choices=["graphs", "reasoner", "filter", "split", "noise", "all"],
                        default=["all"],
                        help="Which steps to run. Default: all")

    args = parser.parse_args()
    dataset_name = args.dataset_name

    # normalize steps
    steps = args.steps
    if "all" in steps:
        steps = ["graphs", "reasoner", "filter", "split", "noise"]

    # Step 1: Graph Construction per Resource
    if "graphs" in steps:
        generate_graphs(dataset_name)

    # Step 2: Run the Reasoner
    if "reasoner" in steps:
        try:
            subprocess.run(
                ["python", "reasoner.py", dataset_name],
                cwd="pellet",  # change working directory to 'pellet'
                check=True
            )
            logger.info("Reasoning completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.info(f"Reasoning failed with error: {e}")

    # Step 3: Filter Inferences
    if "filter" in steps:
        run_filtering(dataset_name)

    # Step 4: Build Train, Test and Validation Graphs
    if "split" in steps:
        train_file, test_file, val_file = build_rdf_datasets(
            dataset_name=dataset_name,
            test_size=0.1,
            seed=1
        )

    # Step 5: Create Noise
    if "noise" in steps:
        experiments = get_experiments(dataset_name)
        add_noise_to_dataset(dataset_name, experiments)