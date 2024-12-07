import os
import pandas as pd

# Directory containing the CSV files
input_dir = "~/projects/motion_signature/data/extracted/"
output_file = "~/projects/motion_signature/data/correlations.csv"

# Expand the home directory path
input_dir = os.path.expanduser(input_dir)
output_file = os.path.expanduser(output_file)

# Initialize a DataFrame to store all correlations
all_correlations = pd.DataFrame()

# Process each CSV file in the directory
for file in os.listdir(input_dir):
    if file.endswith("_features.csv"):  # Check if the file is a features CSV
        file_path = os.path.join(input_dir, file)
        clip_name = os.path.splitext(file)[0]  # Use the file name without extension as clip name

        # Load the CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        # Compute correlations
        try:
            corr_matrix = df.corr()

            # Flatten the correlation matrix into pairs
            corr_pairs = {}
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 != col2:  # Avoid self-correlations
                        corr_pairs[f"{col1}*{col2}"] = corr_matrix.loc[col1, col2]

            # Add to the DataFrame
            all_correlations = pd.concat([all_correlations, pd.DataFrame(corr_pairs, index=[clip_name])])
        except Exception as e:
            print(f"Error processing correlations for {file_path}: {e}")
            continue

# Save the combined correlations to a CSV file
all_correlations.index.name = "clip_name"
all_correlations.to_csv(output_file)

print(f"Correlations saved to {output_file}")
