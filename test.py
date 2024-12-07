import time
from feat import Detector

# Initialize the detector
detector = Detector()

# Path to the video
video_path = "data/dataset/id01026_z_JRShoMw3k_00104.mp4"

# Start timing
start_time = time.time()

# Perform detection
video_prediction = detector.detect(
    video_path, 
    data_type="video", 
    face_detection_threshold=0.7,
    device='cuda',
)

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print elapsed time
print(f"Time taken to process the video: {elapsed_time:.2f} seconds")
