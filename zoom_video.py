import cv2
import numpy as np
import subprocess
import os
import shutil

# ==============================================================================
# --- Private Helper Functions ---
# ==============================================================================

def _find_raw_content_box(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Finds the tightest bounding box around content. Returns None if no content found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    all_points = np.concatenate(contours, axis=0)
    return cv2.boundingRect(all_points)

def _analyze_frame_for_viewport_and_zoom(frame: np.ndarray, margin: int) -> tuple[tuple[int, int, int, int] | None, float]:
    """
    Calculates the final 16:9 viewport and the required zoom multiplier.
    Returns a tuple: (viewport_box (x,y,w,h), zoom_factor).
    """
    video_h, video_w = frame.shape[:2]
    target_aspect_ratio = video_w / video_h if video_h > 0 else 16/9

    raw_box = _find_raw_content_box(frame)
    if raw_box is None:
        return None, 1.0

    x, y, w, h = raw_box
    content_w, content_h = w + (margin * 2), h + (margin * 2)
    content_aspect_ratio = content_w / content_h if content_h > 0 else 1.0

    if content_aspect_ratio > target_aspect_ratio:
        viewport_w = content_w
        viewport_h = viewport_w / target_aspect_ratio
        zoom_factor = video_w / viewport_w
    else:
        viewport_h = content_h
        viewport_w = viewport_h * target_aspect_ratio
        zoom_factor = video_h / viewport_h

    content_center_x, content_center_y = (x + w / 2), (y + h / 2)
    viewport_x = int(content_center_x - viewport_w / 2)
    viewport_y = int(content_center_y - viewport_h / 2)
    
    return (viewport_x, viewport_y, int(viewport_w), int(viewport_h)), zoom_factor

def _draw_debug_results(image: np.ndarray, box_coords: tuple, zoom_val: float, color: tuple, label: str):
    """Helper function to draw the debug box and text on an image."""
    x, y, w, h = box_coords
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)
    text = f"{label} Zoom: {zoom_val:.2f}x"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, 1, 2)
    cv2.putText(image, text, (x + w - text_w - 10, y + text_h + 10), font, 1, (255, 255, 255), 2)

def _render_video_with_zoom(input_path: str, output_path: str, zoom_value: float):
    """Internal helper to render the video with a calculated final zoom level."""
    # ... (This function remains unchanged from the previous version) ...
    print(f"\n--- Starting Video Render Process ---")
    print(f"Applying zoom to finish at {zoom_value:.2f}x")
    output_dir = os.path.dirname(output_path)
    temp_video_file = os.path.join(output_dir, f"temp_{os.path.basename(output_path)}")
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_file, fourcc, fps, (width, height))
    for current_frame in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        start_zoom = 1.0
        zoom_factor = start_zoom + (zoom_value - start_zoom) * (current_frame / (total_frames - 1 if total_frames > 1 else 1))
        crop_width, crop_height = width / zoom_factor, height / zoom_factor
        x, y = (width - crop_width) / 2, (height - crop_height) / 2
        cropped_frame = frame[int(y):int(y + crop_height), int(x):int(x + crop_width)]
        zoomed_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        out.write(zoomed_frame)
        if current_frame > 0 and current_frame % 150 == 0: print(f"  Rendered {current_frame} / {total_frames} frames...")
    cap.release(); out.release()
    print("Video frame rendering complete.")
    print("\n--- Merging video with original audio using FFmpeg ---")
    command = ['ffmpeg', '-y', '-i', temp_video_file, '-i', input_path, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0?', output_path]
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Successfully created final video: {output_path}")
    except subprocess.CalledProcessError as e: print(f"Error during FFmpeg processing:\n{e.stderr.decode()}")
    except FileNotFoundError: print("Error: FFmpeg not found. Please ensure it's in your system's PATH.")
    os.remove(temp_video_file)

# ==============================================================================
# --- Public Function ---
# ==============================================================================

def smart_constant_zoom(input_path: str, output_path: str, margin: int = 50, debug: bool = False):
    """
    Applies a constant zoom to a video, ending with all content perfectly
    framed with a specified margin and a 16:9 aspect ratio.

    Args:
        input_path (str): Path to the source video file.
        output_path (str): Path where the final video will be saved.
        margin (int, optional): The pixel distance to keep from the content on all sides. Defaults to 50.
        debug (bool, optional): If True, saves detection images to the output folder. Defaults to False.
    """
    print(f"--- Analyzing content of '{input_path}' with a {margin}px margin ---")
    if not os.path.exists(input_path): print(f"Error: Input file not found at '{input_path}'"); return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): print(f"Error: Could not open video file '{input_path}'"); return
    ret, first_frame = cap.read()
    cap.release()
    if not ret: print(f"Error: Could not read first frame from '{input_path}'"); return
        
    final_viewport, final_zoom_value = _analyze_frame_for_viewport_and_zoom(first_frame, margin)
    
    if debug:
        print("\n--- Debug mode enabled: generating test images ---")
        output_dir = os.path.dirname(output_path)
        
        # Raw box visualization
        raw_viewport, raw_zoom = _analyze_frame_for_viewport_and_zoom(first_frame, margin=0)
        if raw_viewport:
            image_raw = first_frame.copy()
            _draw_debug_results(image_raw, raw_viewport, raw_zoom, color=(0, 0, 255), label="Raw")
            cv2.imwrite(os.path.join(output_dir, 'debug_raw_viewport.png'), image_raw)
            print(f"Saved raw debug image to '{output_dir}'")

        # Margin box visualization
        if final_viewport:
            image_margin = first_frame.copy()
            _draw_debug_results(image_margin, final_viewport, final_zoom_value, color=(0, 255, 0), label="Margin")
            cv2.imwrite(os.path.join(output_dir, 'debug_margin_viewport.png'), image_margin)
            print(f"Saved margin debug image to '{output_dir}'")
        
    if final_zoom_value <= 1.01:
        print("\nContent (with margin) is larger than the video frame. No zoom will be applied.")
        if input_path != output_path: shutil.copy(input_path, output_path)
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
        shutil.rmtree(OUTPUT_FOLDER); os.makedirs(OUTPUT_FOLDER)

    print("--- Running smart_constant_zoom with debug mode ON ---")
    smart_constant_zoom(
        input_path=INPUT_VIDEO,
        output_path=os.path.join(OUTPUT_FOLDER, 'final_video_smart.mp4'),
        margin=50,
        debug=True  # Set to True to generate the test images
    )
    print("\nProcess finished.")