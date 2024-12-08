import os
import logging
import subprocess

logger = logging.getLogger("compression")

def compress_videos(video_paths, config):
    """
    Compress videos using FFmpeg with specified severity levels.

    Args:
        video_paths (list): List of paths to video files.
        config (dict): Configuration dictionary with compression settings, including `compression_level`.

    Raises:
        ValueError: If compression level is invalid.
        RuntimeError: If compression fails.
    """
    compression_level = config.get("compression_level", 1)

    # Validate compression level
    if compression_level not in [1, 2, 3]:
        logger.error("Invalid compression level. Must be 1, 2, or 3.")
        raise ValueError("Compression level must be 1, 2, or 3.")

    # Define FFmpeg settings for each compression level
    compression_settings = {
        1: {"crf": 23, "preset": "fast"},  # As YouTube/Instagram does
        2: {"crf": 28, "preset": "medium"},  # Three times more compression
        3: {"crf": 33, "preset": "slow"},  # Seven times more compression
    }

    logger.info(f"Starting video compression for {len(video_paths)} files at level {compression_level}...")

    # FFmpeg settings for the chosen level
    crf = compression_settings[compression_level]["crf"]
    preset = compression_settings[compression_level]["preset"]
    TEMP_VIDEO_PATH = "temp_video.mp4"

    for video_path in video_paths:
        if not os.path.isfile(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Compressing video: {video_path} (CRF: {crf}, Preset: {preset})")

        ffmpeg_command = [
            "ffmpeg",
            "-i", video_path,
            "-vcodec", "libx264",
            "-crf", str(crf),  # Compression level
            "-preset", preset,  # Speed vs quality tradeoff
            "-y",  # Overwrite output
            TEMP_VIDEO_PATH
        ]

        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            logger.error(f"FFmpeg compression failed for {video_path}:")
            logger.error(result.stderr.decode("utf-8"))
            raise RuntimeError(f"Compression failed for {video_path}.")

        # Replace the original video with the compressed output
        os.replace(TEMP_VIDEO_PATH, video_path)

    logger.info("Video compression completed for all files.")

if __name__ == "__main__":
    pass
