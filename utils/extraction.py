import os
import cv2
import logging
import subprocess
import mediapipe as mp
import numpy as np
import pandas as pd
from itertools import combinations
from . import normalizators
from yaml import safe_load
from tqdm import tqdm 
from feat import Detector


def extract_features(files: list, config_path: str, correlations: bool = True):
    """
    Extract features from video files by processing each frame within defined intervals (logical clips).

    Args:
        files (list): List of video file paths.
        config_path (str): Path to the configuration file.

    Returns:
        pd.DataFrame: DataFrame containing extracted features for each frame.
    """

    # Load configuration parameters
    with open(config_path, 'r') as config_file:
        config = safe_load(config_file)

    # Load clip configurations
    clip_length = config["clip"]["length"]
    clip_overlap = config["clip"]["overlap"]

    # Whewre to save extracted
    extracted_path = config["extracted_path"]
    
    # If correlations are requested, get the place where they can be stored
    correlations_path = None
    if correlations:
        correlations_path = config["correlations_path"]

    # Load and setup all the extractors requested
    extractors = config["extractors"] # The list of extractors which will be used
    extractors_objects = {}
    for extractor in extractors:
        if extractor is "pyfeat":
            extractors_objects["pyfeat"] = Detector(device='cuda') # careful with batch_size > 1, limited by VRAM
        elif extractor is "mediapipe":
            extractors_objects["mediapipe"] = mp.solutions.pose.Pose(
                                                        static_image_mode=False,
                                                        model_complexity=1,
                                                        enable_segmentation=False,
                                                        min_detection_confidence=0.5,
                                                        min_tracking_confidence=0.5)
        # elif extractor is "...":
        #     extractors_objects["..."] = ...
        # elif extractor is "...":
        #     extractors_objects["..."] = ...
    extractors_features = config["features"]

    # If correlations are requested, get the df where they can be stored
    correlations_df = None
    if correlations:
        correlations_df = pd.DataFrame()

    # Process each video file
    for file in tqdm(files, desc="Processing video files", unit="files"):
        # Load the video and try to read from it
        vid = cv2.VideoCapture(file)
        if not vid.isOpened():
            logging.error(f"Cannot open video file {file}")
            continue

        # Get video properties
        fps = vid.get(cv2.CAP_PROP_FPS)
        vid_frame_num = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get clip properties
        clip_frame_num = int(fps * clip_length)  # Length of the clip in frames
        clip_frame_interval = int(fps * (clip_length - clip_overlap))  # Interval between clip starts

        # Generate list of clip start frames
        clip_starts = []
        current_start_frame = 0
        while current_start_frame + clip_frame_num // 3 < vid_frame_num: # Allow clips of smaller length up to 1/3 of the orig.
            clip_starts.append(current_start_frame)
            current_start_frame = min(current_start_frame + clip_frame_interval, vid_frame_num)

        active_clips = []
        frame_number = 0
        clip_idx = 0

        # Read frames sequentially
        while True:
            ret, frame = vid.read()
            if not ret:
                break  # End of video reached

            # Check if a new clip should start at this frame
            if frame_number in clip_starts:
                clip_index = clip_starts.index(frame_number)
                active_clips.append({
                    'start_frame': frame_number,
                    'frames': [],
                    'idx': None
                })

            # Append the frame to all active clips
            for clip in active_clips:
                clip['frames'].append(frame)

                # Check if the clip has reached the required length
                if len(clip['frames']) == clip_frame_num:
                    # Add a sequential number to a clip, cnt++
                    clip['idx'] = clip_idx; clip_idx += 1
                    clip_name = f"{os.path.splitext(os.path.basename(file))[0]}_c{clip['clip_index']}"

                    output_path = os.path.join(extracted_path, f"{clip_name}_tmp.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
                    out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

                    for frame in clip['frames']:
                        out.write(frame) 
                    out.release()

                    # Extract features from the clip's frames
                    features_extracted = extract_features_clip(output_path, extractors_objects, extractors_features)
                    extracted_features_path = os.path.join(extracted_path, f"{clip_name}_features.csv")
                    features_clip_df.to_csv(extracted_features_path, index=False)

                    # Compute correlation and add them to the df
                    correlations_matrix = features_extracted.corr()

                    # Select required correlations
                    correlations_dict = {}
                    for feature_1, feature_2 in combinations(df_combined.columns, 2):
                        correlation_value = correlations.loc[feature_1, feature_2]
                        correlations_dict[f"{feature_1}*{feature_2}"] = correlation_value

                    # Insert correlations for the video
                    correlations_df[clip_name] = pd.Series(correlations_dict)
                    
                    # Remove the clip from active clips
                    active_clips.remove(clip)

            frame_number += 1

        vid.release()  # Release the video file

    # If correlations are requested, export them
    if correlations:
        correlations_df = correlations_df.T
        correlations_df.to_csv(correlations_path, index=False)

def extract_features_clip(clip_path: str, extractors_objects: list, extractors_features: list) -> pd.DataFrame:
    extracted_features = pd.DataFrame()

    # Extract features using Py-Feat
    if "pyfeat" in extractors_objects.keys():
        pf_extracted_df = pyfeat_extractor(clip_path, extractors_objects["pyfeat"])

        # Specific to py-feat data preprocessing
        used_features = extractors_features["pyfeat"]["used"]
        pf_extracted_df = pf_extracted_df[used_features]

        # Concatenating the final df
        extracted_features = pd.concat([extracted_features, pf_extracted_df], axis=1)
    # Extract features using MEDIAPIPE
    if "mediapipe" in extractors_objects.keys():

        mp_extracted_df =  mediapipe_extractor(clip_path, extractors_objects["mediapipe"], extractors_features["mediapipe"]["all"])
        
        # Specific to mediapipe data preprocessing
        used_features = extractors_features["mediapipe"]["used"]
        mp_extracted_df = mp_extracted_df[used_features]

        # Perform normalisation for the whole df
        mp_extracted_normalised_df = normalize_landmarks_df(mp_extracted_df)

        # Separate X and Y in the coordiantes
        for column in mp_extracted_normalised_df.columns:
            if isinstance(mp_extracted_normalised_df[column].iloc[0], tuple):
                mp_extracted_normalised_df[f"{column}_X"] = mp_extracted_normalised_df[column].apply(lambda coord: coord[0] if coord is not None else None)
                mp_extracted_normalised_df[f"{column}_Y"] = mp_extracted_normalised_df[column].apply(lambda coord: coord[1] if coord is not None else None)
                mp_extracted_normalised_df.drop(column, axis=1, inplace=True)  # Delete the initial column

        # Concatenating the final df
        extracted_features = pd.concat([extracted_features,], axis=1)
    # Extract features using ...
    # if ... in extractors:
    
    return pd.DataFrame(extracted_features)

def pyfeat_extractor(clip_path: str, extractor: Detector) -> feat.Fex:
    return extractor.detect(clip_path, data_type="video",  face_detection_threshold=0.8)

def mediapipe_extractor(clip_path: str, extractor: mp.pose, features: list) -> pd.DataFrame:
    vidObj = cv2.VideoCapture(clip_path)

    if not vidObj.isOpened():
        logging.warning(f"Cannot open video file {clip_path}")
        continue

    landmarks_list = []
    while True:
        ret, frame = vidObj.read()
        if not ret:
            break

        # Convert frame to RGB format as required by MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = mp_pose.process(frame_rgb)
        landmarks = {features[idx]: landmark for idx, landmark in enumerate(results.pose_landmarks.landmark)} if results.pose_landmarks else {feature: None for feature in features}
        landmarks_list.append(landmarks)

    vidObj.release()

    # Create DataFrame from list
    return pd.DataFrame(landmarks_list)

if __name__ == "__main__":
    pass
