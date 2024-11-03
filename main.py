from datasets.frames_processor import process_clips_dataset
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from datasets.clips_extractor import dataset2clips

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXTRACTED_CORRELATIONS_PATH = "datasets/extracted_correlations.csv"
DATASET_PATH = "datasets/dataset"
DATASET_CLIPS_PATH = "datasets/dataset_clips"

if not os.path.isfile(EXTRACTED_CORRELATIONS_PATH):
    logging.info(f"{EXTRACTED_CORRELATIONS_PATH} is not present, building...")
    if not os.path.isdir(DATASET_CLIPS_PATH):
        if not os.path.isdir(DATASET_PATH):
            logging.error("Please DOWNLOAD the DATASET first")
        else:
            dataset2clips(DATASET_PATH)
    else:
        process_clips_dataset(DATASET_CLIPS_PATH)

df = pd.read_csv(EXTRACTED_CORRELATIONS_PATH)

X = df.drop(columns=['is_fake'])
y = df['is_fake']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(), param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

print(classification_report(y_val, y_val_pred))
print(confusion_matrix(y_val, y_val_pred))
