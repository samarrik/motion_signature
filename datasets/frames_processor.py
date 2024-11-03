import logging
import os
import subprocess
from itertools import combinations

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from normalization.normalizators import normalize_landmarks

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MP_BODY_LANDMARKS_DIC = {
    0: "nose",
    1: "leftEyeInner",
    2: "leftEye",
    3: "leftEyeOuter",
    4: "rightEyeInner",
    5: "rightEye",
    6: "rightEyeOuter",
    7: "leftEar",
    8: "rightEar",
    9: "mouthLeft",
    10: "mouthRight",
    11: "leftShoulder",
    12: "rightShoulder",
    13: "leftElbow",
    14: "rightElbow",
    15: "leftWrist",
    16: "rightWrist",
    17: "leftPinky",
    18: "rightPinky",
    19: "leftIndex",
    20: "rightIndex",
    21: "leftThumb",
    22: "rightThumb",
    23: "leftHip",
    24: "rightHip",
    25: "leftKnee",
    26: "rightKnee",
    27: "leftAnkle",
    28: "rightAnkle",
    29: "leftHeel",
    30: "rightHeel",
    31: "leftFootIndex",
    32: "rightFootIndex"
}

MP_NEEDED = [
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist"
]

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


def process_clips_dataset(clips_dataset_path: str = "datasets/dataset_clips") -> pd.DataFrame:
    logging.info("Starting to process clips dataset.")
    EXTRACTED_OF = "datasets/dataset_extracted/extracted_of"
    EXTRACTED_MP = "datasets/dataset_extracted/extracted_mp"
    EXTRACTED_CORRELATIONS = "datasets/extracted_correlations.csv"

    os.makedirs(EXTRACTED_OF, exist_ok=True)
    os.makedirs(EXTRACTED_MP, exist_ok=True)

    if not os.path.isdir(clips_dataset_path):
        logging.error(f"The dataset of clips '{clips_dataset_path}' does not exist.")
        return pd.DataFrame()

    clip_paths = []
    for root, _, files in os.walk(clips_dataset_path):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                clip_paths.append(os.path.join(root, file))

    if not clip_paths:
        logging.error("No video files found in the dataset.")
        return pd.DataFrame()

    logging.info(f"Found {len(clip_paths)} video clip(s) to process.")

    # Run OpenFace extractor on all of the clips
    feature_list = ["-aus", "-pose"]
    command = ['utils/OpenFace/build/bin/FeatureExtraction'] + feature_list
    for clip_path in clip_paths:
        command += ['-f', clip_path]
    command += ["-out_dir", EXTRACTED_OF]
    logging.info("Running OpenFace FeatureExtraction.")
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logging.error(f"OpenFace FeatureExtraction failed: {e}")
        return

    # Remove all generated info files
    for root, _, files in os.walk(EXTRACTED_OF):
        for file in files:
            if file.endswith(".txt"):
                os.remove(os.path.join(root, file))
                logging.info(f"Removed text file: {file} from {EXTRACTED_OF}")

    # Initialize MediaPipe Pose estimator
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)


    # Run MediaPipe extractor on all of the clips
    logging.info("Running MediaPipe landmark extraction.")
    for clip_path in clip_paths:
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
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
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = {
                    MP_BODY_LANDMARKS_DIC[id_l]: (landmark.x, landmark.y)
                    for id_l, landmark in enumerate(results.pose_landmarks.landmark)
                }

                # Normalize pose landmarks
                success, normalized_landmarks = normalize_landmarks(landmarks)

                if not success:
                    landmarks_with_axes = {f"{key}_{axis}": np.nan for key in MP_BODY_LANDMARKS_DIC.values()
                                            for axis in ['X', 'Y']}
                else:
                    landmarks_with_axes = {
                        f"{key}_{axis}": value for key, (x, y) in normalized_landmarks.items()
                        for axis, value in zip(['X', 'Y'], [x, y])
                    }
            else:
                # No pose landmarks detected, add NaNs
                landmarks_with_axes = {f"{key}_{axis}": np.nan for key in MP_BODY_LANDMARKS_DIC.values()
                                        for axis in ['X', 'Y']}

            landmarks_list.append(landmarks_with_axes)

        vidObj.release()

        # Create DataFrame from list
        df_landmarks = pd.DataFrame(landmarks_list)

        # Export the data
        extracted_path = os.path.join(EXTRACTED_MP, f"{clip_name}.csv")
        df_landmarks.to_csv(extracted_path, index=False)

    # Compute correlations
    df_correlations = pd.DataFrame()

    for clip_path in clip_paths:
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        logging.info(f"Computing correlations for clip: {clip_name}")

        try:
            # Load the data extracted using both OpenFace and MediaPipe
            df_of_path = os.path.join(EXTRACTED_OF, f"{clip_name}.csv")
            df_mp_path = os.path.join(EXTRACTED_MP, f"{clip_name}.csv")

            if not os.path.exists(df_of_path) or not os.path.exists(df_mp_path):
                logging.warning(f"Extracted data files for clip {clip_name} are missing. Skipping.")
                continue

            df_of = pd.read_csv(df_of_path)
            df_mp = pd.read_csv(df_mp_path)

            df_combined = pd.concat([df_of, df_mp], axis=1)

            # Drop all the columns which are neither in MP_NEEDED nor in OF_NEEDED
            columns_to_keep = [f"{key}_X" for key in MP_NEEDED] + [f"{key}_Y" for key in MP_NEEDED] + OF_NEEDED
            df_combined = df_combined[columns_to_keep]

            # Drop rows with NaNs
            df_combined.dropna(inplace=True)

            if df_combined.empty:
                logging.warning(f"No valid data to compute correlations for clip: {clip_name}")
                continue

            correlations = df_combined.corr()

            correlations_dict = {}
            for feature_1, feature_2 in combinations(df_combined.columns, 2):
                correlation_value = correlations.loc[feature_1, feature_2]
                correlations_dict[f"{feature_1}*{feature_2}"] = correlation_value

            df_correlations[clip_name] = pd.Series(correlations_dict)

            # Add binary feature for clip name ending with '_real' to df_correlations
            df_correlations.loc['is_fake', clip_name] = 1 if clip_name.endswith('_fake') else 0

        except Exception as e:
            logging.error(f"Failed to compute correlations for clip {clip_name}: {e}")
            continue

    # Transpose the df to make each clip a row, not a column
    df_correlations = df_correlations.T

    # Save the correlations into a CSV file
    try:
        df_correlations.to_csv(EXTRACTED_CORRELATIONS)
        logging.info("Saved correlations to correlations.csv")
    except Exception as e:
        logging.error(f"Failed to save correlations to file: {e}")

    return


if __name__ == "__main__":
    pass
