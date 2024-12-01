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
logger = logging.getLogger("compress")  # Create a logger for this module

def parse_arguments():
    """
    Parse command-line arguments to get the dataset path.
    If no arguments are provided, defaults are used.
    
    Returns:
        argparse.Namespace: Parsed arguments with dataset path and config.
    """

    parser = argparse.ArgumentParser(description="Process video datasets for identity-related tasks.")
    parser.add_argument("--dataset", type=str, default="data/dataset", help="Path to the main folder of the dataset")
    parser.add_argument("--compress_num", type=str, default="0", help="The number of compressing applied")
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

    # Here fill the rest of the code, you need to compress all videos in the dataset exactly the number of times it is written in the argument

if __name__ == "__main__":
    logger.info(f"Parsing the arguments...")
    args = parse_arguments()
    dataset = args.dataset
    compress_num = int(args.compress_num)

    # Validate the dataset directory
    logger.info(f"Validating the dataset '{dataset}'...")
    if not os.path.isdir(dataset):
        logger.error(f"Dataset directory '{dataset}' not found or is invalid.")
        raise FileNotFoundError(f"{dataset} is not a valid directory.")
    
    # Ensure the directory contains only video files by checking file extensions
    files = [os.path.join(dataset, f) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))]
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.3gp')

    # Check if all files have valid extensions
    invalid_files = [f for f in files if not f.lower().endswith(valid_extensions)]
    if invalid_files:
        logger.error(f"The dataset directory '{dataset}' contains invalid files: {invalid_files}")
        raise ValueError(f"The dataset directory '{dataset}' must contain only video files.")

    # Process each video file in the dataset
    logger.info(f"Starting video compression for all files in '{dataset}'...")
    for video_file in files:
        try:
            logger.info(f"Compressing video: {video_file} with {compress_num} iterations.")
            compress_video(video_file, compress_num)
        except Exception as e:
            logger.error(f"Failed to compress video '{video_file}': {str(e)}")
    
    logger.info("Video compression completed for all files.")
