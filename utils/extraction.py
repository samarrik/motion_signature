
import os
import logging
import pandas as pd
from itertools import combinations
from yaml import safe_load
from tqdm import tqdm
from feat import Detector
from feat.utils.io import video_to_tensor

# Set up logging
logger  = logging.getLogger("extraction")

def extract_features(files: list, config_path: str):
    """
    Extract features from video files by processing each frame within defined intervals (logical clips).

    Args:
        files (list): List of video file paths.
        config_path (str): Path to the configuration file.
    Returns:
        None
    """
    # Load configuration parameters
    with open(config_path, 'r') as config_file:
        config = safe_load(config_file)

    # Load the names of all the extractors requested
    extractors = config["extractors"]

    # Configure the extractors
    extractors_objects = {}
    if "py-feat" in extractors:
        extractors_objects["py-feat"] = Detector(device="cuda")

    # Define the path where the raw extracted data will be saved
    extracted_path = config['extracted_path']

    for file in files:
        try:
            # Generate features path
            video_name = os.path.splitext(os.path.basename(file))[0]
            features_path = os.path.join(extracted_path, f"{video_name}.csv")

            # Compute the prediction
            detector = extractors_objects.get("py-feat")
            video_prediction = detector.detect(
                video_to_tensor(file), 
                data_type="tensor",
                face_detection_threshold=0.8,
                num_workers=10,
                batch_size=500,
                save=features_path,
            )

        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            continue

def compute_correlations(config):
    # Get the folder where all the raw extracted data is stored
    pass

# def extract_features_clip(clip_path: str, extractors_objects: dict, extractors_features: dict) -> pd.DataFrame:
#     """
#     Extract features from a video clip using specified extractors.

#     Args:
#         clip_path (str): Path to the video clip.
#         extractors_objects (dict): Dictionary of initialized extractor objects.
#         extractors_features (dict): Dictionary specifying features to use from each extractor.

#     Returns:
#         pd.DataFrame: DataFrame containing extracted features.
#     """
#     extracted_features = pd.DataFrame()

#     # Extract features using pyAFAR
#     if "pyafar" in extractors_objects:
#         pf_extracted_df = pyafar_extractor(
#             clip_path, 
#             extractors_objects["pyafar"], 
#             None,)

#         # Specific to pyAFAR data preprocessing
#         used_features = extractors_features["pyafar"]["used"]
#         pf_extracted_df = pf_extracted_df[used_features]

#         # Concatenate to the final DataFrame
#         extracted_features = pd.concat([extracted_features, pf_extracted_df], axis=1)

#     # Extract features using MediaPipe
#     if "mediapipe" in extractors_objects:
#         mp_extracted_df = mediapipe_extractor(
#             clip_path,
#             extractors_objects["mediapipe"],
#             extractors_features["mediapipe"]["all"]
#         )

#         # Specific to MediaPipe data preprocessing
#         used_features = extractors_features["mediapipe"]["used"]
#         mp_extracted_df = mp_extracted_df[used_features]

#         # Perform normalization for the whole DataFrame
#         mp_extracted_normalized_df = normalizators.normalize_landmarks_df(mp_extracted_df)

#         # Separate X and Y coordinates
#         for column in mp_extracted_normalized_df.columns:
#             if isinstance(mp_extracted_normalized_df[column].iloc[0], tuple):
#                 mp_extracted_normalized_df[f"{column}_X"] = mp_extracted_normalized_df[column].apply(
#                     lambda coord: coord[0] if coord is not None else None)
#                 mp_extracted_normalized_df[f"{column}_Y"] = mp_extracted_normalized_df[column].apply(
#                     lambda coord: coord[1] if coord is not None else None)
#                 mp_extracted_normalized_df.drop(column, axis=1, inplace=True)  # Delete the initial column

#         # Concatenate to the final DataFrame
#         extracted_features = pd.concat([extracted_features, mp_extracted_normalized_df], axis=1)
#     # Add other extractors here if needed

#     # Handle NaN values, replace with mean if the number of NaNs is less then 30% of the values in the column
#     for column in extracted_features.columns:
#         nan_ratio = extracted_features[column].isna().mean()
#         if nan_ratio < 0.5:
#             # Replace NaN with column mean
#             extracted_features[column].fillna(extracted_features[column].mean(), inplace=True)
#         # Else leave NaNs as they are

#     return extracted_features


# def pyafar_extractor(file: str, extractor: dict, features: None = None) -> pd.DataFrame:
#     """
#     Extract features using pyAFAR.

#     Args:
#         file (str): Path to the video file.
#         extractor (None): is not used
#         features (None): is not used
#     Returns:
#         pd.DataFrame: DataFrame containing extracted features with frame numbers.
#     """
#     # Process the entire video
#     result = adult_afar(
#         filename=file,  
#         AUs=extractor['AUs'], 
#         GPU=extractor['GPU'], 
#         max_frames=extractor['max_frames'], 
#         AU_Int=extractor['AU_Int'], 
#         batch_size=extractor['batch_size'], 
#         PID=extractor['PID'])
#     df = pd.DataFrame.from_dict(result)
#     return df

# def mediapipe_extractor(clip_path: str, extractor: mp.solutions.pose.Pose, features: list) -> pd.DataFrame:
#     """
#     Extract features using MediaPipe Pose.

#     Args:
#         clip_path (str): Path to the video clip.
#         extractor (mp.solutions.pose.Pose): Initialized MediaPipe Pose extractor.
#         features (list): List of feature names.

#     Returns:
#         pd.DataFrame: DataFrame containing extracted features.
#     """
#     vid_obj = cv2.VideoCapture(clip_path)

#     if not vid_obj.isOpened():
#         logging.warning(f"Cannot open video file {clip_path}")
#         return pd.DataFrame(columns=features)

#     landmarks_list = []
#     while True:
#         ret, frame = vid_obj.read()
#         if not ret:
#             break
#         # Convert frame to RGB format as required by MediaPipe
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         results = extractor.process(frame_rgb)
#         if results.pose_landmarks:
#             landmarks = {features[idx]: (landmark.x, landmark.y) for idx, landmark in enumerate(results.pose_landmarks.landmark)}
#         else:
#             landmarks = {feature: None for feature in features}
#         landmarks_list.append(landmarks)

#     vid_obj.release()

#     # Create DataFrame from list
#     return pd.DataFrame(landmarks_list)

if __name__ == "__main__":
    pass
