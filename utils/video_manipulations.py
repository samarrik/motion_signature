import os
import shutil
import subprocess
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip

# Set up logging
logger  = logging.getLogger("video_manipulations")

def videos2clips(videos_path: str, clip_length: int = 10, overlap: int = 5):

    def _video2clips(input_file: str):
        with VideoFileClip(input_file) as video:
            duration = video.duration
            start_time, clip_index = 0.0, 0

            while start_time < duration:
                end_time = min(start_time + clip_length, duration)
                clip = video.subclip(start_time, end_time)

                relative_path = os.path.relpath(os.path.dirname(input_file), dataset_clips_path)
                clip_name = f"{os.path.splitext(os.path.basename(input_file))[0]}_clip_{clip_index}.mp4" # {vid num}_{id of POI}_{group}_clip_{clip index}
                clip_dir = os.path.join(dataset_clips_path, relative_path)
                os.makedirs(clip_dir, exist_ok=True)
                clip_path = os.path.join(clip_dir, clip_name)

                logging.info(f"Creating clip: {clip_path}")
                clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

                start_time += clip_length - overlap
                clip_index += 1

        os.remove(input_file)

    clips_path = os.path.join(os.path.dirname(videos_path), "clips")

    if os.path.exists(clips_path):
        shutil.rmtree(clips_path)
    shutil.copytree(videos_path, clips_path)

    for root, _, files in os.walk(dataset_clips_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                file_path = os.path.join(root, file)
                _video2clips(file_path, 10, 5)

if __name__ == "__main__":
    pass
    