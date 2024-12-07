import os
import logging
import subprocess

logger = logging.getLogger("compression")

def compress_videos(video_paths, config):
    """
    Compress videos using FFmpeg.

    Args:
        video_paths (list): List of paths to video files.
        config (dict): Configuration dictionary with compression settings.

    Raises:
        ValueError: If compression settings are invalid.
        RuntimeError: If compression fails.
    """
    compressions_num = config.get("compressions_num", 1)

    if compressions_num < 1:
        logger.error("The number of compressions must be an int and at least 1.")
        raise ValueError("The number of compressions must be an int and at least 1.")

    logger.info(f"Starting video compression for {len(video_paths)} files with {compressions_num} passes...")

    TEMP_VIDEO_PATH = "temp_video.mp4"

    for video_path in video_paths:
        if not os.path.isfile(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_path_current = video_path

        for i in range(compressions_num):
            
            ffmpeg_command = [
                "ffmpeg",
                "-i", video_path_current,
                "-vcodec", "libx264",
                "-crf", "28",  # Compression level (lower = better quality)
                "-preset", "fast",  # Speed vs quality tradeoff
                "-y",  # Overwrite output
                TEMP_VIDEO_PATH
            ]

            result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg compression failed for {video_path} on pass {i + 1}:")
                logger.error(result.stderr.decode("utf-8"))
                raise RuntimeError(f"Compression failed for {video_path}.")

            # Update the current video path for the next compression pass
            if i < compressions_num - 1:
                os.replace(TEMP_VIDEO_PATH, video_path_current)

        # Replace the original video with the final compressed output
        os.replace(TEMP_VIDEO_PATH, video_path)

    logger.info("Video compression completed for all files.")


if __name__ == "__main__":
    pass
