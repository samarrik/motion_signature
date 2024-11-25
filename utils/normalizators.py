# Adaptation of Matyas Bohacek's code, all rights belong to Matyas Bohacek 2022-present
# https://github.com/matyasbohacek/spoter/blob/main/normalization/hand_normalization.py

import logging

def normalize_landmarks(landmarks: dict) -> (bool, dict):    
    valid_landmarks = True

    # Prevent from even starting the analysis if some necessary elements are not present
    if (landmarks["leftShoulder"][0] == 0 or landmarks["rightShoulder"][0] == 0 or landmarks["leftEye"][0] == 0):
        logging.info("Cannot normalize the frame, crucial landmarks are missing")
        for key in landmarks.keys():
            landmarks[key] = (None, None)  # Set all landmarks to None
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
                logging.warning("Problematic normalization")
                for key in landmarks.keys():
                    landmarks[key] = (None, None)  # Set all landmarks to None
                break
            
            landmark_norm_x = (landmarks[key][0] - starting_point[0]) / (ending_point[0] - starting_point[0])
            landmark_norm_y = (landmarks[key][1] - starting_point[0]) / (ending_point[0] - starting_point[0])
            landmarks[key] = landmark_norm_x, landmark_norm_y

    return (True, landmarks)

import pandas as pd


def normalize_landmarks_df(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    original_landmarks = landmarks_df.copy()

    # Normalize individual landmarks for each row
    for index, row in landmarks_df.iterrows():
        # Prevent from even starting the analysis if some necessary elements are not present
        if (row["leftShoulder"][0] == 0 or row["rightShoulder"][0] == 0 or row["leftEye"][0] == 0):
            logging.info(f"Cannot normalize the frame at index {index}, crucial landmarks are missing, NULLing the frame")
            landmarks_df.at[index, :] = (None, None)
            continue  # Skip to the next row

        left_shoulder = (row["leftShoulder"][0], row["leftShoulder"][1])
        right_shoulder = (row["rightShoulder"][0], row["rightShoulder"][1])

        # Calculation of the estimation of the head metric
        shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)
        head_metric = shoulder_distance / 2

        # Set the starting and ending point of the normalization bounding box
        starting_point = [row["leftShoulder"][0] - 2 * head_metric,
                          row["leftEye"][0] + head_metric]
        ending_point = [row["rightShoulder"][0] + 2 * head_metric, starting_point[1] - 6 * head_metric]

        # Normalize individual landmarks for the current row
        for key in landmarks_df.columns:
            # Prevent from trying to normalize incorrectly captured points
            if row[key] == (0, 0):
                continue

            if (ending_point[0] - starting_point[0]) == 0 or (starting_point[1] - ending_point[1]) == 0:
                logging.warning(f"Problematic normalization at index {index}, NULLing the frame")
                landmarks_df.at[index, :] = (None, None)
                break
            
            landmark_norm_x = (row[key][0] - starting_point[0]) / (ending_point[0] - starting_point[0])
            landmark_norm_y = (row[key][1] - starting_point[1]) / (ending_point[1] - starting_point[1])
            landmarks_df.at[index, key] = (landmark_norm_x, landmark_norm_y)

    return landmarks_df

if __name__ == "__main__":
    pass
