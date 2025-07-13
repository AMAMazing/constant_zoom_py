import cv2
import numpy as np
import subprocess
import os
import shutil
from typing import Literal

def process_video_with_zoom(
    input_path: str,
    output_path: str,
    mode: Literal['endpoint', 'rate'],
    zoom_value: float,
    keep_temp_files: bool = False
):
    """
    Processes a video to apply a continuous, centered zoom effect.

    Args:
        input_path (str): Path to the source video file.
        output_path (str): Path where the final video will be saved.
        mode (Literal['endpoint', 'rate']): The zoom behavior.
            - 'endpoint': The video will finish at the zoom_value. (e.g., a zoom_value of 1.5
              means the video will end at 150% zoom, regardless of video length).
            - 'rate': The zoom increases at a constant rate. (Not yet implemented).
        zoom_value (float): The numeric value for the zoom mode. For 'endpoint', this is the
                           final zoom multiplier (e.g., 1.1 for 10%, 2.0 for 100%).
        keep_temp_files (bool, optional): If True, the temporary silent video file will not be
                                          deleted. Defaults to False.
    """
    print(f"--- Starting Video Process ---")
    print(f"Mode: '{mode}', Zoom Value: {zoom_value}")

    # --- Video Processing Setup ---
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    # Create a temporary file path in the same directory as the output
    output_dir = os.path.dirname(output_path)
    temp_video_file = os.path.join(output_dir, f"temp_{os.path.basename(output_path)}")

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Error: Could not read frames from '{input_path}'. It may be corrupt or empty.")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_file, fourcc, fps, (width, height))
    print(f"Video Properties: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames.")

    # --- Frame-by-Frame Processing Loop ---
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_zoom = 1.0
        zoom_factor = 1.0

        # --- Calculate Zoom Based on Mode ---
        if mode == 'endpoint':
            # Linear interpolation from 1.0 to the target zoom_value over the video's duration
            zoom_factor = start_zoom + (zoom_value - start_zoom) * (current_frame / total_frames)
        
        elif mode == 'rate':
            # This logic would be different, based on time (t) not percentage of completion
            # For example: zoom_factor = 1.0 + zoom_value * (current_frame / fps)
            raise NotImplementedError("The 'rate' mode is not yet implemented.")
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'endpoint' or 'rate'.")

        # Calculate the dimensions and position of the crop
        crop_width = width / zoom_factor
        crop_height = height / zoom_factor
        x = (width - crop_width) / 2
        y = (height - crop_height) / 2
        
        # Crop the frame and resize it back to full size
        cropped_frame = frame[int(y):int(y + crop_height), int(x):int(x + crop_width)]
        zoomed_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        out.write(zoomed_frame)
        
        current_frame += 1
        if current_frame > 0 and current_frame % 150 == 0:
            print(f"  Processed {current_frame} / {total_frames} frames...")

    cap.release()
    out.release()
    print("Video frame processing complete.")

    # --- Merge with Audio using FFmpeg ---
    print("\n--- Merging video with original audio using FFmpeg ---")
    command = [
        'ffmpeg', '-y',
        '-i', temp_video_file,
        '-i', input_path,
        '-c:v', 'copy', '-c:a', 'copy',
        '-map', '0:v:0', '-map', '1:a:0?',
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully created final video: {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error during FFmpeg processing:")
        print(e.stderr.decode())
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")

    # --- Cleanup ---
    if not keep_temp_files:
        print("\n--- Cleaning up temporary files ---")
        try:
            os.remove(temp_video_file)
            print(f"Removed temporary file: {temp_video_file}")
        except OSError as e:
            print(f"Error removing temporary file: {e}")

    print("\nProcess finished.")


# ==============================================================================
# --- Main Execution Block (This is where you run the script) ---
# ==============================================================================
if __name__ == "__main__":
    
    # --- Setup ---
    INPUT_VIDEO = 'input.mp4'
    OUTPUT_FOLDER = 'output'

    # Clean and prepare the output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    else:
        print(f"Cleaning output folder: '{OUTPUT_FOLDER}'...")
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # --- Run the Test Case ---
    # We will test the 'endpoint' mode. The video will end at 150% (1.5x) zoom.
    process_video_with_zoom(
        input_path=INPUT_VIDEO,
        output_path=os.path.join(OUTPUT_FOLDER, 'final_video_endpoint.mp4'),
        mode='endpoint',
        zoom_value=1.1  # This means 150% final zoom. For 10% zoom, use 1.1.
    )