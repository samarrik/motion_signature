"""
main.py

Processes a video dataset for identity-related tasks by dividing videos into clips and extracting specified features.

Main functionalities:
- Checks if the dataset exists and its valid
- Runs extraction

Usage:
    python3 main.py --dataset path/to/dataset
"""

import os
import sys
import logging
import argparse
from utils.extraction import extract_features

# Configure logging for better readability and context
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")  # Create a logger for this module

def parse_arguments():
    """
    Parse command-line arguments to get the dataset path.
    If no arguments are provided, defaults are used.
    
    Returns:
        argparse.Namespace: Parsed arguments with dataset path and config.
    """

    parser = argparse.ArgumentParser(description="Process video datasets for identity-related tasks.")
    parser.add_argument("--dataset", type=str, default="data/dataset", help="Path to the main folder of the dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info(f"Parsing the arguments...")
    args = parse_arguments()
    dataset = args.dataset

    # Check if the provided dataset path is a valid directory
    logger.info(f"Validating the dataset '{dataset}'...")
    if not os.path.isdir(dataset):
        logger.error(f"Dataset directory '{dataset}' not found or is invalid.")
        raise FileNotFoundError(f"{dataset} is not a valid directory.")
    
    # Ensure the directory contains only video files by checking file extensions
    files = [os.path.join(dataset, f) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))]
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.3gp')
    
    # Check if all files have valid extensions
    if not all(f.lower().endswith(valid_extensions) for f in files):
        logger.error(f"The dataset directory '{dataset}' contains invalid files.")
        raise ValueError(f"The dataset directory '{dataset}' must contain only video files.")

    # Check the config file
    config = args.config
    if not os.path.isfile(config):
        logger.error(f"Config file '{config}' not found.")
        raise FileNotFoundError(f"The config file '{config}' does not exist.")

    logger.info(f"Dataset '{dataset}' has been checked and is valid.")
    logger.info("Starting the extraction process...")

    # Convert each video to clips and extract features
    extract_features(files, config, correlations=True)
    logger.info("Feature extraction completed successfully.")
