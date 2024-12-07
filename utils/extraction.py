import os
import logging
import pandas as pd
from itertools import combinations
from yaml import safe_load
from tqdm import tqdm
from feat import Detector
from feat.utils.io import video_to_tensor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("extraction")


def load_extractors(extractors_config):
    """
    Load and configure feature extractors.

    Args:
        extractors_config (list): List of extractor names.

    Returns:
        dict: Dictionary of initialized extractor objects.
    """
    extractors_objects = {}

    if "py-feat" in extractors_config:
        try:
            logger.info("Initializing py-feat detector...")
            extractors_objects["py-feat"] = Detector(device="cuda")
        except Exception as e:
            logger.error(f"Failed to initialize py-feat detector: {e}")
            raise RuntimeError("py-feat initialization failed.")

    return extractors_objects


def extract_features(files, config):
    """
    Extract features from video files by processing each frame within defined intervals (logical clips).

    Args:
        files (list): List of video file paths.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    # Validate configuration
    extractors = config.get("extractors", [])
    if not extractors:
        logger.error("No extractors specified in the configuration.")
        raise ValueError("The configuration must specify at least one extractor.")

    extracted_features_raw_path = config.get("extracted_features_raw_path")
    if not extracted_features_raw_path:
        logger.error("No extracted path specified in the configuration.")
        raise ValueError("The configuration must specify an extracted path.")
    
    # Ensure the output directory exists
    os.makedirs(extracted_features_raw_path, exist_ok=True)

    # Initialize extractors
    extractors_objects = load_extractors(extractors)

    # Process each video file
    for file in tqdm(files, desc="Processing Videos"):
        try:
            video_name = os.path.splitext(os.path.basename(file))[0]
            features_path = os.path.join(extracted_path, f"{video_name}.csv")

            logger.info(f"Processing video: {file}")
            detector = extractors_objects.get("py-feat")

            # Ensure detector is configured before processing
            if not detector:
                logger.error("Py-feat detector not configured.")
                raise RuntimeError("Feature extraction requires a configured py-feat detector.")

            # Perform feature detection
            video_tensor = video_to_tensor(file)
            video_prediction = detector.detect(
                video_tensor,
                data_type="tensor",
                face_detection_threshold=0.8,
                num_workers=10,
                batch_size=500,
                save=features_path,
            )

            logger.info(f"Features saved to: {features_path}")

        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            continue

if __name__ == "__main__":
    pass
