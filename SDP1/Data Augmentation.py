import cv2
import os
import imgaug.augmenters as iaa
# Configuration
video_dir = r"D:\Dataset Final\Yasmine"  # Directory containing videos
output_dir = r"D:\Dataset Final\Y output"  # Output directory for frames
num_frames_to_extract = 18  # Number of frames to extract
num_augmentations = 1  # Number of augmented frames per original frame
# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Data augmentation sequence
augmenters = iaa.SomeOf((1, 3), [
    iaa.Fliplr(0.5),  # Horizontal flip
    iaa.Affine(rotate=(-15, 15)),  # Random rotation
    iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
    iaa.AdditiveGaussianNoise(scale=(10, 30)),  # Add noise
    iaa.GaussianBlur(sigma=(0.0, 3.0)),  # Blur
])
 
# Process all video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
frame_id = 0  # Global frame counter to ensure unique filenames
for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames_to_extract, 1)
    print(f"Processing: {video_file}, Total frames: {total_frames}, Interval: {interval}")
    frame_count = 0
    extracted_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or extracted_frames >= num_frames_to_extract:
            break
        # Extract frames at regular intervals
        if frame_count % interval == 0:
            # Save original frame with a unique name
            original_frame_path = os.path.join(output_dir, f"frame_{frame_id}.jpg")
            cv2.imwrite(original_frame_path, frame)
            # Generate augmented frames
            for i in range(num_augmentations):
                augmented_frame = augmenters(image=frame)
                augmented_frame_path = os.path.join(output_dir, f"frame_{frame_id}_aug_{i}.jpg")
                cv2.imwrite(augmented_frame_path, augmented_frame)
            extracted_frames += 1
            frame_id += 1
        frame_count += 1
    cap.release()
    print(f"{extracted_frames} frames (with augmentations) extracted from {video_file} and saved to {output_dir}")
print(f"All videos processed. Frames saved to {output_dir}")
