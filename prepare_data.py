import os
import sys
import logging
import argparse
from pathlib import Path
from yaml import safe_load
from utils import compress_videos, extract_features, compute_correlations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")


def load_config(config_path):
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Parsed configuration file.
    """
    try:
        with open(config_path, 'r') as file:
            return safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        sys.exit(1)


def validate_dataset(dataset_path):
    """
    Validate the dataset directory structure and contents.

    Args:
        dataset_path (str): Path to the dataset directory.

    Raises:
        ValueError: If dataset is invalid.
    """
    dataset_dir = Path(dataset_path)

    if not dataset_dir.is_dir():
        logger.error(f"Dataset directory '{dataset_path}' does not exist or is invalid.")
        sys.exit(1)

    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.3gp')
    invalid_files = [
        file for file in dataset_dir.iterdir()
        if file.is_file() and not file.suffix.lower() in valid_extensions
    ]

    if invalid_files:
        logger.error(f"Invalid files found in dataset directory: {[file.name for file in invalid_files]}")
        sys.exit(1)

    logger.info(f"Dataset '{dataset_path}' validated successfully.")


def process_videos(dataset_path, config):
    """
    Perform feature extraction and correlation computation.

    Args:
        dataset_path (str): Path to the dataset directory.
        config (dict): Configuration settings.
    """
    files = [str(f) for f in Path(dataset_path).glob('*') if f.is_file()]

    # Optional: Video compression
    if config.get("compress_videos", False):
        logger.info("Starting the video compression process...")
        compress_videos(files, config)
        logger.info("Video compression completed successfully.")
    
    # Optional: Feature Extraction
    if config.get("extract_features", False):
        logger.info("Starting the feature extraction process...")
        extract_features(files, config)
        logger.info("Feature extraction completed successfully.")

    # Optional: Compute Correlations
    if config.get("compute_correlations", False):
        logger.info("Starting the correlation computation process...")
        compute_correlations(config)
        logger.info("Correlation computation completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Process a video dataset for feature extraction.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load dataset path
    dataset_path = config.get("dataset_path", None)
    if not dataset_path:
        logger.error("Dataset path must be provided in the configuration.")
        sys.exit(1)

    # Validate dataset
    validate_dataset(dataset_path)

    # Process features
    process_videos(dataset_path, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)
