
import os
import logging

from utils.extractors import correlation_extractor
from utils.video_manipulations import videos2clips

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    
    CORRELATIONS_PATH = os.path.join("data", "correlations.csv")
    VIDEOS_PATH = os.path.join("data", "videos")
    CLIPS_PATH = os.path.join("data", "clips")

    if not os.path.isfile(CORRELATIONS_PATH):
        logging.info(f"{CORRELATIONS_PATH} is not present, making an attempt to extract...")

        if not os.path.isdir(CLIPS_PATH):
            logging.info(f"{CLIPS_PATH} is not present, making an attempt to yield clips from videos...")

            if not os.path.isdir(VIDEOS_PATH):
                logging.error(f"{VIDEOS_PATH} is not present, cannot neither yield clips nor extract correlations")
                exit(1)

            else:
                videos2clips(VIDEOS_PATH)

        else:
            correlation_extractor(CLIPS_PATH)

