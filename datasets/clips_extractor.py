import os
import shutil
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def dataset2clips(dataset_path: str = "datasets/dataset"):
    """
    Creates a copy of the dataset and converts each video into smaller overlapping clips.

    :param dataset_path: Path to the dataset
    """
    dataset_clips_path = f"{dataset_path}_clips"

    def video2clips(input_file: str, clip_length: int = 10, overlap: int = 5):
        """
        Cuts the selected video into clips of specified length and overlap.

        :param input_file: Path to the video file
        :param clip_length: Length of each clip in seconds
        :param overlap: Overlap between clips in seconds
        """
        with VideoFileClip(input_file) as video:
            duration = video.duration
            start_time, clip_index = 0.0, 0

            while start_time < duration:
                end_time = min(start_time + clip_length, duration)
                clip = video.subclip(start_time, end_time)

                relative_path = os.path.relpath(os.path.dirname(input_file), dataset_clips_path)
                clip_name = f"{os.path.splitext(os.path.basename(input_file))[0]}_clip_{clip_index}.mp4"
                clip_dir = os.path.join(dataset_clips_path, relative_path)
                os.makedirs(clip_dir, exist_ok=True)
                clip_path = os.path.join(clip_dir, clip_name)

                logging.info(f"Creating clip: {clip_path}")
                clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

                start_time += clip_length - overlap
                clip_index += 1

        os.remove(input_file)

    if os.path.exists(dataset_clips_path):
        shutil.rmtree(dataset_clips_path)
    shutil.copytree(dataset_path, dataset_clips_path)

    for root, _, files in os.walk(dataset_clips_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                file_path = os.path.join(root, file)
                video2clips(file_path)

if __name__ == "__main__":
    pass
    