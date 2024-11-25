import os
import cv2
import logging
import mediapipe as mp
import pandas as pd
from itertools import combinations
from yaml import safe_load
from tqdm import tqdm
from feat import Detector
from . import normalizators


def extract_features(files: list, config_path: str, correlations: bool = True):
    """
    Extract features from video files by processing each frame within defined intervals (logical clips).

    Args:
        files (list): List of video file paths.
        config_path (str): Path to the configuration file.
        correlations (bool): Whether to compute and save correlations.

    Returns:
        None
    """
    # Load configuration parameters
    with open(config_path, 'r') as config_file:
        config = safe_load(config_file)

    # Load clip configurations
    clip_length = config["clip"]["length"]
    clip_overlap = config["clip"]["overlap"]

    # Where to save extracted features
    extracted_path = config["extracted_path"]

    # If correlations are requested, get the path where they can be stored
    correlations_path = config.get("correlations_path") if correlations else None

    # Load and set up all the extractors requested
    extractors = config["extractors"]  # The list of extractors to be used
    extractors_objects = {}
    for extractor in extractors:
        if extractor == "pyfeat":
            extractors_objects["pyfeat"] = Detector(device='cuda')  # Careful with batch_size > 1, limited by VRAM
        elif extractor == "mediapipe":
            extractors_objects["mediapipe"] = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        # Add other extractors here if needed

    extractors_features = config["features"]

    # DataFrame to store correlations if requested
    correlations_df = pd.DataFrame() if correlations else None

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
        while current_start_frame + clip_frame_num // 3 < vid_frame_num:  # Allow clips of smaller length up to 1/3 of the original
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
                active_clips.append({
                    'start_frame': frame_number,
                    'frames': [],
                    'idx': None
                })

            # Append the frame to all active clips
            for clip in active_clips[:]:
                clip['frames'].append(frame)

                # Check if the clip has reached the required length
                if len(clip['frames']) == clip_frame_num:
                    # Assign a sequential number to the clip
                    clip['idx'] = clip_idx
                    clip_idx += 1
                    clip_name = f"{os.path.splitext(os.path.basename(file))[0]}_c{clip['idx']}"

                    output_path = os.path.join(extracted_path, f"{clip_name}_tmp.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
                    out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

                    for clip_frame in clip['frames']:
                        out.write(clip_frame)
                    out.release()

                    # Extract features from the clip's frames
                    features_extracted = extract_features_clip(output_path, extractors_objects, extractors_features)
                    extracted_features_path = os.path.join(extracted_path, f"{clip_name}_features.csv")
                    features_extracted.to_csv(extracted_features_path, index=False)

                    # Compute correlation and add them to the DataFrame if requested
                    if correlations:
                        correlations_matrix = features_extracted.corr()

                        # Select required correlations
                        correlations_dict = {}
                        for feature_1, feature_2 in combinations(features_extracted.columns, 2):
                            correlation_value = correlations_matrix.loc[feature_1, feature_2]
                            correlations_dict[f"{feature_1}*{feature_2}"] = correlation_value

                        # Insert correlations for the video
                        correlations_df = correlations_df.append(pd.Series(correlations_dict, name=clip_name))

                    # Remove the clip from active clips
                    active_clips.remove(clip)

            frame_number += 1

        vid.release()  # Release the video file

    # If correlations are requested, export them
    if correlations and not correlations_df.empty:
        correlations_df.to_csv(correlations_path, index=True)


def extract_features_clip(clip_path: str, extractors_objects: dict, extractors_features: dict) -> pd.DataFrame:
    """
    Extract features from a video clip using specified extractors.

    Args:
        clip_path (str): Path to the video clip.
        extractors_objects (dict): Dictionary of initialized extractor objects.
        extractors_features (dict): Dictionary specifying features to use from each extractor.

    Returns:
        pd.DataFrame: DataFrame containing extracted features.
    """
    extracted_features = pd.DataFrame()

    # Extract features using Py-Feat
    if "pyfeat" in extractors_objects:
        pf_extracted_df = pyfeat_extractor(clip_path, extractors_objects["pyfeat"])

        # Specific to py-feat data preprocessing
        used_features = extractors_features["pyfeat"]["used"]
        pf_extracted_df = pf_extracted_df[used_features]

        # Concatenate to the final DataFrame
        extracted_features = pd.concat([extracted_features, pf_extracted_df], axis=1)

    # Extract features using MediaPipe
    if "mediapipe" in extractors_objects:
        mp_extracted_df = mediapipe_extractor(
            clip_path,
            extractors_objects["mediapipe"],
            extractors_features["mediapipe"]["all"]
        )

        # Specific to MediaPipe data preprocessing
        used_features = extractors_features["mediapipe"]["used"]
        mp_extracted_df = mp_extracted_df[used_features]

        # Perform normalization for the whole DataFrame
        mp_extracted_normalized_df = normalizators.normalize_landmarks_df(mp_extracted_df)

        # Separate X and Y coordinates
        for column in mp_extracted_normalized_df.columns:
            if isinstance(mp_extracted_normalized_df[column].iloc[0], tuple):
                mp_extracted_normalized_df[f"{column}_X"] = mp_extracted_normalized_df[column].apply(
                    lambda coord: coord[0] if coord is not None else None)
                mp_extracted_normalized_df[f"{column}_Y"] = mp_extracted_normalized_df[column].apply(
                    lambda coord: coord[1] if coord is not None else None)
                mp_extracted_normalized_df.drop(column, axis=1, inplace=True)  # Delete the initial column

        # Concatenate to the final DataFrame
        extracted_features = pd.concat([extracted_features, mp_extracted_normalized_df], axis=1)

    # Add other extractors here if needed

    return extracted_features


def pyfeat_extractor(clip_path: str, extractor: Detector) -> pd.DataFrame:
    """
    Extract features using Py-Feat.

    Args:
        clip_path (str): Path to the video clip.
        extractor (Detector): Initialized Py-Feat detector.

    Returns:
        pd.DataFrame: DataFrame containing extracted features.
    """
    return extractor.detect(clip_path, data_type="video", face_detection_threshold=0.8)


def mediapipe_extractor(clip_path: str, extractor: mp.solutions.pose.Pose, features: list) -> pd.DataFrame:
    """
    Extract features using MediaPipe Pose.

    Args:
        clip_path (str): Path to the video clip.
        extractor (mp.solutions.pose.Pose): Initialized MediaPipe Pose extractor.
        features (list): List of feature names.

    Returns:
        pd.DataFrame: DataFrame containing extracted features.
    """
    vid_obj = cv2.VideoCapture(clip_path)

    if not vid_obj.isOpened():
        logging.warning(f"Cannot open video file {clip_path}")
        return pd.DataFrame(columns=features)

    landmarks_list = []
    while True:
        ret, frame = vid_obj.read()
        if not ret:
            break

        # Convert frame to RGB format as required by MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = extractor.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = {features[idx]: (landmark.x, landmark.y) for idx, landmark in enumerate(results.pose_landmarks.landmark)}
        else:
            landmarks = {feature: None for feature in features}
        landmarks_list.append(landmarks)

    vid_obj.release()

    # Create DataFrame from list
    return pd.DataFrame(landmarks_list)


if __name__ == "__main__":
    pass
