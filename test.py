import time
from feat import Detector
from feat.utils.io import video_to_tensor

# Initialize the detector
detector = Detector(device='cuda')

# Path to the video
video_path = "data/dataset/id01026_z_JRShoMw3k_00104.mp4"

# Start timing
start_time = time.time()

# Perform detection
video_prediction = detector.detect(
    video_to_tensor(video_path), 
    data_type="tensor",
    face_detection_threshold=0.8,
    num_workers=10,
    batch_size=500,
)

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print elapsed time
print(f"Time taken to process the video: {elapsed_time:.2f} seconds")
