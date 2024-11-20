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

OF_NEEDED = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "pose_Rx",
    "pose_Rz"
    # mouthH(x_54-x_48), mouthV(x_51-x_57) have to be normalised, but author does not mention it, so I skip them
]


def extract_features(files: list, config_path: str):
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

                    # Remove the clip from active clips
                    active_clips.remove(clip)

            frame_number += 1

        vid.release()  # Release the video file

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

        # Normalise the features
        if results.pose_landmarks:
            landmarks = {
                MP_BODY_LANDMARKS_DIC[id_l]: (landmark.x, landmark.y) for id_l, landmark in enumerate(results.pose_landmarks.landmark)
                }

            # Normalize pose landmarks
            success, normalized_landmarks = normalizators.normalize_landmarks(landmarks)

            if not success:
                landmarks_with_axes = {
                    f"{key}_{axis}": np.nan for key in MP_BODY_LANDMARKS_DIC.values() for axis in ['X', 'Y']
                    }
            else:
                landmarks_with_axes = {
                    f"{key}_{axis}": value for key, (x, y) in normalized_landmarks.items() for axis, value in zip(['X', 'Y'], [x, y])
                    }
        else:
            # No pose landmarks detected, add NaNs
            landmarks_with_axes = {
                f"{key}_{axis}": np.nan for key in MP_BODY_LANDMARKS_DIC.values() for axis in ['X', 'Y']
                }
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
        landmarks = {features[idx]: landmark for idx, landmark in enumerate(results.pose_landmarks.landmark)}
        landmarks_list.append(landmarks)

    vidObj.release()

    # Create DataFrame from list
    return pd.DataFrame(landmarks_list)


# def correlation_extractor(clips_dataset_path: str) -> pd.DataFrame:
    
#     EXTRACTED_OF = os.path.join("data", "extracted", "extracted_of")
#     EXTRACTED_MP = os.path.join("data", "extracted", "extracted_mp")
#     CORRELATIONS = os.path.join("data", "correlations.csv")

#     os.makedirs(EXTRACTED_OF, exist_ok=True)
#     os.makedirs(EXTRACTED_MP, exist_ok=True)

#     # Collect all the paths to the clips
#     clip_paths = []
#     for root, _, files in os.walk(clips_dataset_path):
#         for file in files:
#             if file.lower().endswith((".mp4", ".avi", ".mov")):
#                 clip_paths.append(os.path.join(root, file))

#     if not clip_paths:
#         logging.error("No video files found in the dataset.")
#         return
    
#     else:
#         logging.info(f"Found {len(clip_paths)} video clip(s) to process.")
        
#     if not os.path.exists(EXTRACTED_OF):
#         # Run OpenFace extractor on all of the clips
#         feature_list = ["-aus", "-pose"]
#         command = ['utils/OpenFace/build/bin/FeatureExtraction'] + feature_list
#         for clip_path in clip_paths:
#             command += ['-f', clip_path]
#         command += ["-out_dir", EXTRACTED_OF]

#         logging.info("Running OpenFace FeatureExtraction.")
#         try:
#             subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         except subprocess.CalledProcessError as e:
#             logging.error(f"OpenFace FeatureExtraction failed: {e}")
#             return 

#         # Remove all generated info files
#         for root, _, files in os.walk(EXTRACTED_OF):
#             for file in files:
#                 if file.endswith(".txt"):
#                     os.remove(os.path.join(root, file))
#                     logging.info(f"Removed text file: {file} from {EXTRACTED_OF}")



#     if not os.path.exists(EXTRACTED_MP):
#         # Initialize MediaPipe Pose estimator
#         mp_pose = mp.solutions.pose.Pose(
#             static_image_mode=False,
#             model_complexity=1,
#             enable_segmentation=False,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5)

#         # Run MediaPipe extractor on all of the clips
#         logging.info("Running MediaPipe landmark extraction.")
#         for clip_path in clip_paths:
#             clip_name = os.path.splitext(os.path.basename(clip_path))[0]
#             vidObj = cv2.VideoCapture(clip_path)

#             if not vidObj.isOpened():
#                 logging.warning(f"Cannot open video file {clip_path}")
#                 continue

#             landmarks_list = []
#             while True:
#                 ret, frame = vidObj.read()
#                 if not ret:
#                     break

#                 # Convert frame to RGB format as required by MediaPipe
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = mp_pose.process(frame_rgb)

#                 if results.pose_landmarks:
#                     landmarks = {
#                         MP_BODY_LANDMARKS_DIC[id_l]: (landmark.x, landmark.y) for id_l, landmark in enumerate(results.pose_landmarks.landmark)
#                         }

#                     # Normalize pose landmarks
#                     success, normalized_landmarks = normalizators.normalize_landmarks(landmarks)

#                     if not success:
#                         landmarks_with_axes = {
#                             f"{key}_{axis}": np.nan for key in MP_BODY_LANDMARKS_DIC.values() for axis in ['X', 'Y']
#                             }
#                     else:
#                         landmarks_with_axes = {
#                             f"{key}_{axis}": value for key, (x, y) in normalized_landmarks.items() for axis, value in zip(['X', 'Y'], [x, y])
#                             }
#                 else:
#                     # No pose landmarks detected, add NaNs
#                     landmarks_with_axes = {
#                         f"{key}_{axis}": np.nan for key in MP_BODY_LANDMARKS_DIC.values() for axis in ['X', 'Y']
#                         }

#                 landmarks_list.append(landmarks_with_axes)

#             vidObj.release()

#             # Create DataFrame from list
#             df_landmarks = pd.DataFrame(landmarks_list)

#             # Export the data
#             extracted_path = os.path.join(EXTRACTED_MP, f"{clip_name}.csv")
#             df_landmarks.to_csv(extracted_path, index=False)

#     # Compute correlations
#     df_correlations = pd.DataFrame()

#     for clip_path in clip_paths:
#         clip_name = os.path.splitext(os.path.basename(clip_path))[0]
#         logging.info(f"Computing correlations for clip: {clip_name}")

#         try:
#             # Load the data extracted using both OpenFace and MediaPipe
#             df_of_path = os.path.join(EXTRACTED_OF, f"{clip_name}.csv")
#             df_mp_path = os.path.join(EXTRACTED_MP, f"{clip_name}.csv")

#             if not os.path.exists(df_of_path) or not os.path.exists(df_mp_path):
#                 logging.warning(f"Extracted data files for clip {clip_name} are missing. Skipping.")
#                 continue

#             df_of = pd.read_csv(df_of_path)
#             df_mp = pd.read_csv(df_mp_path)

#             df_combined = pd.concat([df_of, df_mp], axis=1)

#             # Drop all the columns which are neither in MP_NEEDED nor in OF_NEEDED
#             columns_to_keep = [f"{key}_X" for key in MP_NEEDED] + [f"{key}_Y" for key in MP_NEEDED] + OF_NEEDED
#             df_combined = df_combined[columns_to_keep]

#             correlations = df_combined.corr()

#             correlations_dict = {}
#             for feature_1, feature_2 in combinations(df_combined.columns, 2):
#                 correlation_value = correlations.loc[feature_1, feature_2]
#                 correlations_dict[f"{feature_1}*{feature_2}"] = correlation_value

#             df_correlations[clip_name] = pd.Series(correlations_dict)

#         except Exception as e:
#             logging.error(f"Failed to compute correlations for clip {clip_name}: {e}")
#             continue

#     # Transpose the df to make each clip a row, not a column
#     df_correlations = df_correlations.T

#     # Save the correlations into a CSV file
#     try:
#         df_correlations.to_csv(CORRELATIONS, index_label='clip_name')
#         logging.info("Saved correlations to correlations.csv")
#     except Exception as e:
#         logging.error(f"Failed to save correlations to file: {e}")

if __name__ == "__main__":
    pass
