import cv2
import numpy as np
import subprocess
import os
import shutil

def _find_raw_content_box(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Finds the tightest bounding box around content. Returns None if no content found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    all_points = np.concatenate(contours, axis=0)
    return cv2.boundingRect(all_points)

def _analyze_frame_for_final_zoom(frame: np.ndarray, margin: int) -> float:
    """
    Calculates the final zoom multiplier needed to frame the content with a 16:9 aspect ratio.
    """
    video_h, video_w = frame.shape[:2]
    # Ensure we don't divide by zero if height is 0
    target_aspect_ratio = video_w / video_h if video_h > 0 else 16/9

    raw_box = _find_raw_content_box(frame)
    if raw_box is None:
        print("Warning: No content detected. No zoom will be applied.")
        return 1.0

    x, y, w, h = raw_box

    # Create the desired content area by adding the margin
    content_w = w + (margin * 2)
    content_h = h + (margin * 2)

    # Determine the zoom factor based on which dimension is the constraint
    content_aspect_ratio = content_w / content_h if content_h > 0 else 1.0

    if content_aspect_ratio > target_aspect_ratio:
        # Content is WIDER than the target frame, so width dictates the zoom
        zoom_factor = video_w / content_w
    else:
        # Content is TALLER than the target frame, so height dictates the zoom
        zoom_factor = video_h / content_h

    return zoom_factor

def _render_video_with_zoom(input_path: str, output_path: str, zoom_value: float):
    """Internal helper to render the video with a calculated final zoom level."""
    print(f"\n--- Starting Video Render Process ---")
    print(f"Applying zoom to finish at {zoom_value:.2f}x")

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
        # Linearly interpolate the zoom for the current frame
        zoom_factor = start_zoom + (zoom_value - start_zoom) * (current_frame / (total_frames -1))
        
        crop_width, crop_height = width / zoom_factor, height / zoom_factor
        x, y = (width - crop_width) / 2, height / 2 - crop_height / 2
        
        cropped_frame = frame[int(y):int(y + crop_height), int(x):int(x + crop_width)]
        zoomed_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        out.write(zoomed_frame)
        
        if current_frame > 0 and current_frame % 150 == 0:
            print(f"  Rendered {current_frame} / {total_frames} frames...")

    cap.release()
    out.release()
    print("Video frame rendering complete.")

    # Merge with Audio using FFmpeg
    print("\n--- Merging video with original audio using FFmpeg ---")
    command = [ 'ffmpeg', '-y', '-i', temp_video_file, '-i', input_path, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0?', output_path ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully created final video: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg processing:\n{e.stderr.decode()}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")

    # Cleanup temporary file
    os.remove(temp_video_file)

def smart_constant_zoom(input_path: str, output_path: str, margin: int = 50):
    """
    Applies a constant zoom to a video, ending with all content perfectly
    framed with a specified margin and a 16:9 aspect ratio.
    """
    # --- Smart Analysis Step ---
    print(f"--- Analyzing content of '{input_path}' with a {margin}px margin ---")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'"); return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'"); return
        
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: Could not read first frame from '{input_path}'"); return
        
    # Calculate the required zoom value
    final_zoom_value = _analyze_frame_for_final_zoom(first_frame, margin)
    
    # --- Video Processing Step ---
    if final_zoom_value <= 1.01: # Use a small tolerance
        print("Content (with margin) is larger than the video frame. No zoom will be applied.")
        if input_path != output_path:
            shutil.copy(input_path, output_path)
        return
        
    _render_video_with_zoom(input_path, output_path, final_zoom_value)


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
if __name__ == "__main__":
    
    INPUT_VIDEO = 'input.mp4'
    OUTPUT_FOLDER = 'output'

    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    else:
        print(f"Cleaning output folder: '{OUTPUT_FOLDER}'...")
        shutil.rmtree(OUTPUT_FOLDER)
        os.makedirs(OUTPUT_FOLDER)

    # --- Run the smart zoom function ---
    smart_constant_zoom(
        input_path=INPUT_VIDEO,
        output_path=os.path.join(OUTPUT_FOLDER, 'final_video_smart.mp4'),
        margin=50
    )
    print("\nProcess finished.")