import cv2
import numpy as np
import subprocess
import os
import shutil

# --- Configuration ---
input_file = 'input.mp4'
output_folder = 'output'  # The name for our organized output directory

start_zoom = 1.0  # Start with no zoom (1.0x)
end_zoom = 1.5    # End at 1.5x zoom

# --- Step 0: Setup and Clean the Output Directory ---
print(f"--- Step 0: Setting up output directory: '{output_folder}' ---")

# Create the directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: '{output_folder}'")
else:
    # If it does exist, clean it out for a fresh run
    print(f"Directory '{output_folder}' already exists. Cleaning its contents...")
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Define file paths *inside* the output folder
final_output_file = os.path.join(output_folder, 'output_final.mp4')
temp_video_file = os.path.join(output_folder, 'temp_video_no_audio.mp4')

# --- Step 1: Process the video with OpenCV ---
print("\n--- Step 1: Processing video frames with OpenCV ---")
cap = cv2.VideoCapture(input_file)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_video_file, fourcc, fps, (width, height))

print(f"Video Properties: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames.")
print(f"Outputting silent video to: {temp_video_file}")

current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate zoom and crop
    zoom_factor = start_zoom + (end_zoom - start_zoom) * (current_frame / total_frames)
    crop_width = width / zoom_factor
    crop_height = height / zoom_factor
    x = (width - crop_width) / 2
    y = (height - crop_height) / 2
    
    # Crop and resize frame
    cropped_frame = frame[int(y):int(y + crop_height), int(x):int(x + crop_width)]
    zoomed_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    out.write(zoomed_frame)
    
    current_frame += 1
    if current_frame > 0 and current_frame % 100 == 0:
        print(f"Processed {current_frame} / {total_frames} frames...")

cap.release()
out.release()
print("OpenCV processing complete.")

# --- Step 2: Merge video with original audio using FFmpeg ---
print("\n--- Step 2: Merging video with original audio using FFmpeg ---")

command = [
    'ffmpeg', '-y',
    '-i', temp_video_file,
    '-i', input_file,
    '-c:v', 'copy',
    '-c:a', 'copy',
    '-map', '0:v:0',
    '-map', '1:a:0?', # Added '?' to make audio optional, preventing errors on silent inputs
    final_output_file
]

try:
    # Using Popen to better handle stdout/stderr for potentially long processes
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("Error during FFmpeg processing:")
        print(stderr)
    else:
        print(f"Successfully created final video: {final_output_file}")
except FileNotFoundError:
    print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")
except Exception as e:
    print(f"An error occurred: {e}")


# --- Step 3: Cleanup ---
print("\n--- Step 3: Cleaning up temporary files ---")
try:
    os.remove(temp_video_file)
    print(f"Removed temporary file: {temp_video_file}")
except OSError as e:
    print(f"Error removing temporary file: {e}")

print("\nAll done!")