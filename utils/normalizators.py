# Adaptation of Matyas Bohacek's code, all rights belong to Matyas Bohacek 2022-present
# https://github.com/matyasbohacek/spoter/blob/main/normalization/hand_normalization.py

import logging

def normalize_landmarks(landmarks: dict) -> (bool, dict):    
    valid_landmarks = True
    original_landmarks = landmarks

    # Prevent from even starting the analysis if some necessary elements are not present
    if (landmarks["leftShoulder"][0] == 0 or landmarks["rightShoulder"][0] == 0 or landmarks["leftEye"][0] == 0):
        logging.info("Cannot normalize the frame, crucial landmarks are missing")
        valid_landmarks = False
    else:
        left_shoulder = (landmarks["leftShoulder"][0], landmarks["leftShoulder"][0])
        right_shoulder = (landmarks["rightShoulder"][1], landmarks["rightShoulder"][1])

        # Calculation of the estimation of the head metric
        # https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf
        shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)
        head_metric = shoulder_distance / 2

        # Set the starting and ending point of the normalization bounding box
        starting_point = [landmarks["leftShoulder"][0] - 2 * head_metric,
                          landmarks["leftEye"][0] + head_metric]
        ending_point = [landmarks["rightShoulder"][0] + 2 * head_metric, starting_point[1] - 6 * head_metric]

        # Ensure that all of the bounding-box-defining coordinates are not out of the picture (is not used to generalise better)
        # if starting_point[0] < 0: starting_point[0] = 0
        # if starting_point[1] < 0: starting_point[1] = 0
        # if ending_point[0] < 0: ending_point[0] = 0
        # if ending_point[1] < 0: ending_point[1] = 0

        # Normalize individual landmarks and save the results
        for key in landmarks:
            # Prevent from trying to normalize incorrectly captured points
            if landmarks[key] == (0, 0):
                continue

            if (ending_point[0] - starting_point[0]) == 0 or (starting_point[1] - ending_point[1]) == 0:
                logging.info("Problematic normalization")
                break
            
            landmark_norm_x = (landmarks[key][0] - starting_point[0]) / (ending_point[0] - starting_point[0])
            landmark_norm_y = (landmarks[key][1] - starting_point[0]) / (ending_point[0] - starting_point[0])
            landmarks[key] = landmark_norm_x, landmark_norm_y

    if valid_landmarks:
        return (True, landmarks)
    else:
        return (False, original_landmarks)

import pandas as pd

def normalize_landmarks_df(landmarks_df: pd.DataFrame) -> (bool, pd.DataFrame):
    valid_landmarks = True
    original_landmarks = landmarks_df.copy()

    # Normalize individual landmarks for each row
    for index, row in landmarks_df.iterrows():
        # Prevent from even starting the analysis if some necessary elements are not present
        if (row["leftShoulder"] == 0 or row["rightShoulder"] == 0 or row["leftEye"] == 0):
            logging.info(f"Cannot normalize the frame at index {index}, crucial landmarks are missing")
            valid_landmarks = False
            continue  # Skip to the next row

        left_shoulder = (row["leftShoulder"], row["leftShoulder"])
        right_shoulder = (row["rightShoulder"], row["rightShoulder"])

        # Calculation of the estimation of the head metric
        shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)
        head_metric = shoulder_distance / 2

        # Set the starting and ending point of the normalization bounding box
        starting_point = [row["leftShoulder"] - 2 * head_metric,
                          row["leftEye"] + head_metric]
        ending_point = [row["rightShoulder"] + 2 * head_metric, starting_point[1] - 6 * head_metric]

        # Normalize individual landmarks for the current row
        for key in landmarks_df.columns:
            # Prevent from trying to normalize incorrectly captured points
            if row[key] == (0, 0):
                continue

            if (ending_point[0] - starting_point[0]) == 0 or (starting_point[1] - ending_point[1]) == 0:
                logging.info(f"Problematic normalization at index {index}")
                valid_landmarks = False
                break
            
            landmark_norm_x = (row[key] - starting_point[0]) / (ending_point[0] - starting_point[0])
            landmark_norm_y = (row[key] - starting_point[1]) / (ending_point[1] - starting_point[1])
            landmarks_df.at[index, key] = (landmark_norm_x, landmark_norm_y)

    if valid_landmarks:
        return (True, landmarks_df)
    else:
        return (False, original_landmarks)

if __name__ == "__main__":
    pass
