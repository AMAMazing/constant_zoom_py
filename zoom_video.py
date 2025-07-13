import cv2
import numpy as np
import subprocess
import os
import shutil
from typing import Optional

def _find_content_bounding_box(frame: np.ndarray, margin: int, bg_color: Optional[np.ndarray] = None) -> tuple[int, int, int, int]:
    """
    Analyzes a frame to find a single bounding box that encloses all non-background content.

    Returns:
        A tuple (x, y, w, h) for the bounding box.
    """
    # 1. Convert to Grayscale for thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Automatically detect background color from top-left corner if not provided
    if bg_color is None:
        bg_color = frame[0, 0]

    # Simple assumption: background is a dark color.
    # We find a threshold to separate dark from light. A more robust method could use color distance.
    # Here, we'll use Otsu's method which is great for bimodal images (like black text on white bg, or vice versa)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If the background is light (e.g., white), the content is black. We need to invert the mask.
    if np.mean(bg_color) > 127:
        thresh = cv2.bitwise_not(thresh)

    # 3. Find contours of all content "blobs"
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No content found, return the full frame dimensions
        h, w = frame.shape[:2]
        return 0, 0, w, h

    # 4. Combine all contours into a single master bounding box
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)

    # 5. Add the user-defined margin
    x_margin = max(0, x - margin)
    y_margin = max(0, y - margin)
    w_margin = min(frame.shape[1], w + (margin * 2))
    h_margin = min(frame.shape[0], h + (margin * 2))

    return x_margin, y_margin, w_margin, h_margin


def _process_video_with_endpoint_zoom(
    input_path: str,
    output_path: str,
    zoom_value: float,
    keep_temp_files: bool = False
):
    """Internal helper to process a video with a calculated final zoom level."""
    print(f"--- Starting Video Process ---")
    print(f"Final Zoom Target: {zoom_value:.2f}x")

    output_dir = os.path.dirname(output_path)
    temp_video_file = os.path.join(output_dir, f"temp_{os.path.basename(output_path)}")
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_file, fourcc, fps, (width, height))

    # Frame-by-frame processing loop
    for current_frame in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        start_zoom = 1.0
        zoom_factor = start_zoom + (zoom_value - start_zoom) * (current_frame / total_frames)
        
        crop_width, crop_height = width / zoom_factor, height / zoom_factor
        x, y = (width - crop_width) / 2, (height - crop_height) / 2
        
        cropped_frame = frame[int(y):int(y + crop_height), int(x):int(x + crop_width)]
        zoomed_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        out.write(zoomed_frame)
        
        if current_frame > 0 and current_frame % 150 == 0:
            print(f"  Processed {current_frame} / {total_frames} frames...")

    cap.release()
    out.release()
    print("Video frame processing complete.")

    # Merge with Audio
    print("\n--- Merging video with original audio using FFmpeg ---")
    command = [ 'ffmpeg', '-y', '-i', temp_video_file, '-i', input_path, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0?', output_path ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully created final video: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg processing:\n{e.stderr.decode()}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")

    if not keep_temp_files:
        os.remove(temp_video_file)


def smart_constant_zoom(input_path: str, output_path: str, margin: int = 20):
    """
    Applies a constant zoom to a video, ending with all content perfectly
    framed with a specified margin.

    Args:
        input_path (str): Path to the source video file.
        output_path (str): Path where the final video will be saved.
        margin (int, optional): The pixel distance to keep from the content on all sides. Defaults to 20.
    """
    # --- Smart Analysis Step ---
    print(f"--- Analyzing content of '{input_path}' with a {margin}px margin ---")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return
        
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from '{input_path}'")
        cap.release()
        return
        
    video_height, video_width = first_frame.shape[:2]

    # 1. Find the bounding box of the content
    x, y, w, h = _find_content_bounding_box(first_frame, margin)
    print(f"Detected content box (with margin): x={x}, y={y}, w={w}, h={h}")

    # 2. Calculate the required zoom factor to fit this box
    # We need to zoom enough to satisfy both width and height constraints
    zoom_w = video_width / w
    zoom_h = video_height / h
    final_zoom_value = max(zoom_w, zoom_h) # Take the larger zoom factor to ensure everything fits

    cap.release()
    
    # --- Video Processing Step ---
    if final_zoom_value <= 1.0:
        print("Content (with margin) is larger than the video frame. No zoom will be applied.")
        # If no zoom is needed, we can just copy the file
        if input_path != output_path:
            shutil.copy(input_path, output_path)
        return
        
    _process_video_with_endpoint_zoom(input_path, output_path, final_zoom_value)


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
if __name__ == "__main__":
    
    INPUT_VIDEO = 'input.mp4'
    OUTPUT_FOLDER = 'output'

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    else:
        print(f"Cleaning output folder: '{OUTPUT_FOLDER}'...")
        shutil.rmtree(OUTPUT_FOLDER)
        os.makedirs(OUTPUT_FOLDER)

    # --- Run the Test Case ---
    # We will use the new "smart" function.
    smart_constant_zoom(
        input_path=INPUT_VIDEO,
        output_path=os.path.join(OUTPUT_FOLDER, 'final_video_smart.mp4'),
        margin=50  # Let's use a generous 50px margin
    )