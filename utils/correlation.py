import os
import pandas as pd

def compute_correlations(config):

    # Use already extracted

    # Use only requested features

    # Compute correlations in all combinations (described b4)
    
    pass



# # Directory containing the CSV files
# input_dir = "~/projects/motion_signature/data/extracted/"
# output_file = "~/projects/motion_signature/data/correlations.csv"

# # Expand the home directory path
# input_dir = os.path.expanduser(input_dir)
# output_file = os.path.expanduser(output_file)

# # Initialize a DataFrame to store all correlations
# all_correlations = pd.DataFrame()

# # Process each CSV file in the directory
# for file in os.listdir(input_dir):
#     if file.endswith("_features.csv"):  # Check if the file is a features CSV
#         file_path = os.path.join(input_dir, file)
#         clip_name = os.path.splitext(file)[0]  # Use the file name without extension as clip name

#         # Load the CSV
#         try:
#             df = pd.read_csv(file_path)
#         except Exception as e:
#             print(f"Error loading {file_path}: {e}")
#             continue

#         # Compute correlations
#         try:
#             corr_matrix = df.corr()

#             # Flatten the correlation matrix into pairs
#             corr_pairs = {}
#             for col1 in corr_matrix.columns:
#                 for col2 in corr_matrix.columns:
#                     if col1 != col2:  # Avoid self-correlations
#                         corr_pairs[f"{col1}*{col2}"] = corr_matrix.loc[col1, col2]

#             # Add to the DataFrame
#             all_correlations = pd.concat([all_correlations, pd.DataFrame(corr_pairs, index=[clip_name])])
#         except Exception as e:
#             print(f"Error processing correlations for {file_path}: {e}")
#             continue

# # Save the combined correlations to a CSV file
# all_correlations.index.name = "clip_name"
# all_correlations.to_csv(output_file)

# print(f"Correlations saved to {output_file}")



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
