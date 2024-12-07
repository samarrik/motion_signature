import os
import sys
import logging
import subprocess
import argparse

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


def compress_video(input_path, times=1):
    """
    Compress a video using FFmpeg to mimic YouTube-like compression.

    Args:
        input_path (str): Path to the input video file.
        times (int): Number of times to recompress the video.

    Returns:
        None
    """
    if times < 1:
        logging.error("The number of compressions must be at least 1.")
        raise ValueError("The number of compressions must be at least 1.")
    
    temp_path = "temp_video.mp4"
    current_input = input_path
    for i in range(times):        
        # Compress video using FFmpeg
        ffmpeg_command = [
            "ffmpeg",
            "-i", current_input,
            "-vcodec", "libx264",
            "-crf", "28",  # Compression level (lower = better quality)
            "-preset", "fast",  # Speed vs quality tradeoff
            "-y",  # Overwrite output
            temp_path
        ]
        
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            logging.error(f"FFmpeg compression failed: {result.stderr.decode('utf-8')}")
            raise RuntimeError("Compression failed.")
        
        if i < times - 1:
            os.replace(temp_path, current_input)
    
    # Replace the original file with the final compressed output
    os.replace(temp_path, input_path)
    logging.info(f"Compression completed successfully for {input_path}.")

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
