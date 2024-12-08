import os
import logging
import pandas as pd
from tqdm import tqdm

# Constants
CONSTANT_VIDEO_SIZE_S = 20

logger = logging.getLogger("correlations")


def compute_correlations(config):
    """
    Compute correlation matrices for video features from extracted CSV files.
    
    For each video, the function splits the feature data into logical clips
    (with optional overlap) and computes correlation matrices for each clip.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Validate configuration
    extracted_features_raw_path = config.get("extracted_features_raw_path")
    if not extracted_features_raw_path:
        logger.error("No 'extracted_features_raw_path' specified in the configuration.")
        raise ValueError("The configuration must specify 'extracted_features_raw_path'.")
    
    correlations_path = config.get("correlations_path")
    if not correlations_path:
        logger.error("No 'correlations_path' specified in the configuration.")
        raise ValueError("The configuration must specify 'correlations_path'.")

    extractors = config.get("extractors")
    if not extractors:
        logger.error("No extractors specified in the configuration.")
        raise ValueError("The configuration must specify 'extractors'.")

    extractors_features = config.get("extractors_features", {})
    clips = config.get("clips")
    if not clips:
        logger.error("No clip configuration specified in the configuration.")
        raise ValueError("The configuration must specify 'clips'.")

    # Ensure output directory exists
    os.makedirs(correlations_path, exist_ok=True)

    # List all CSV files
    extracted_features_files = [
        os.path.join(extracted_features_raw_path, f)
        for f in os.listdir(extracted_features_raw_path)
        if f.lower().endswith('.csv')
    ]

    # If no CSV files found, log and return
    if not extracted_features_files:
        logger.warning("No CSV files found in the extracted_features_raw_path.")
        return

    # Determine the selected features from the configured extractors
    selected_features = []
    for extractor in extractors:
        features = extractors_features.get(extractor, [])
        selected_features.extend(features)

    # Process each clip configuration
    for _, clips_config in clips.items():
        print(clips_config)
        length_clip = clips_config.get("length", 20)
        overlap_clip = clips_config.get("overlap", 10)
        full_clip = clips_config.get("full", True)

        # Prepare a DataFrame to store all correlations for this configuration
        correlations = pd.DataFrame()

        # Process each extracted feature file
        logger.info(f"Processing clip configuration: l{length_clip}_o{overlap_clip}_f{int(full_clip)}")
        for extracted_file in tqdm(extracted_features_files, desc="Computing Correlations"):
            # File's basename
            video_basename = os.path.splitext(os.path.basename(extracted_file))[0]

            # Load the file into a DataFrame
            df_video = pd.read_csv(extracted_file)

            # Filter to selected features if provided
            if selected_features:
                existing_features = [col for col in selected_features if col in df_video.columns]
                if not existing_features:
                    logger.warning(f"No selected features found in file {extracted_file}. Skipping.")
                    continue
                df_video = df_video[existing_features]

            # Get the video's properties
            frame_cnt_video = len(df_video)
            fps_video = frame_cnt_video / CONSTANT_VIDEO_SIZE_S

            # Compute clip size in frames
            frame_cnt_clip = int(round(fps_video * length_clip))
            frame_int_clip = int(round(fps_video * (length_clip - overlap_clip)))

            # Generate a list of clip starts
            clip_starts = []
            current_start_frame = 0
            if full_clip:
                # Extract full-length overlapping clips
                while current_start_frame + frame_cnt_clip <= frame_cnt_video:
                    clip_starts.append(current_start_frame)
                    current_start_frame = min(current_start_frame + frame_int_clip, frame_cnt_video)
            else:
                # If not full clips, take just one clip from the start if possible
                if frame_cnt_clip <= frame_cnt_video:
                    clip_starts.append(current_start_frame)

            # Process clips sequentially
            active_clips = []
            frame_number = 0
            clip_idx = 0

            while frame_number < frame_cnt_video:
                if frame_number in clip_starts:
                    active_clips.append({
                        'start_frame': frame_number,
                        'frames': [],
                        'idx': None
                    })

                # Append the frame to all active clips
                for clip in active_clips[:]:
                    clip['frames'].append(frame_number)

                    # Check if the clip has reached the required length
                    if len(clip['frames']) == frame_cnt_clip:
                        clip['idx'] = clip_idx
                        clip_idx += 1

                        # Extract the clip's frames
                        df_clip = df_video.iloc[clip['frames']]

                        # Compute the correlation matrix
                        corr_matrix = df_clip.corr()

                        # Flatten the correlation matrix into pairs, excluding self-correlations
                        corr_pairs = {
                            f"{col1}*{col2}": corr_matrix.loc[col1, col2]
                            for col1 in corr_matrix.columns
                            for col2 in corr_matrix.columns
                            if col1 != col2
                        }

                        # Add to the DataFrame with a unique clip name
                        clip_label = f"{video_basename}_c{clip_idx:05d}"
                        correlations = pd.concat([correlations, pd.DataFrame(corr_pairs, index=[clip_label])])

                        # Remove the completed clip
                        active_clips.remove(clip)

                frame_number += 1

        # Save the computed correlations for this clip configuration
        output_filename = f"correlations_l{length_clip}_o{overlap_clip}_f{int(full_clip)}.csv"
        correlations_name = os.path.join(correlations_path, output_filename)

        try:
            correlations.to_csv(correlations_name)
            logger.info(f"Correlations saved to: {correlations_name}")
        except Exception as e:
            logger.error(f"Failed to save correlations file '{correlations_name}': {e}")
            raise

if __name__ == "__main__":
    pass
