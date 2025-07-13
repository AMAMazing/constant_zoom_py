import cv2
import numpy as np
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


def analyze_frame_for_zoom(frame: np.ndarray, margin: int) -> tuple[tuple[int, int, int, int], float]:
    """
    Calculates the final 16:9 viewport and the required zoom multiplier.

    Returns:
        A tuple containing:
        - The viewport bounding box (x, y, w, h) with a 16:9 aspect ratio.
        - The calculated zoom factor (float).
    """
    video_h, video_w = frame.shape[:2]
    target_aspect_ratio = video_w / video_h

    # 1. Find the tight bounding box of the actual content
    raw_box = _find_raw_content_box(frame)
    if raw_box is None:
        print("Warning: No content detected. Defaulting to full frame (1.0x zoom).")
        return (0, 0, video_w, video_h), 1.0

    x, y, w, h = raw_box

    # 2. Create the desired content area by adding the margin
    # This is the box we want to perfectly frame
    content_w = w + (margin * 2)
    content_h = h + (margin * 2)
    content_x = x - margin
    content_y = y - margin
    
    # 3. Calculate the final viewport box that is 16:9
    content_aspect_ratio = content_w / content_h

    if content_aspect_ratio > target_aspect_ratio:
        # Content is WIDER than the target frame, so width is the constraint
        viewport_w = content_w
        viewport_h = viewport_w / target_aspect_ratio
        zoom_factor = video_w / viewport_w
    else:
        # Content is TALLER than the target frame, so height is the constraint
        viewport_h = content_h
        viewport_w = viewport_h * target_aspect_ratio
        zoom_factor = video_h / viewport_h

    # Center the final viewport around the original content center
    content_center_x = content_x + content_w / 2
    content_center_y = content_y + content_h / 2
    
    viewport_x = int(content_center_x - viewport_w / 2)
    viewport_y = int(content_center_y - viewport_h / 2)

    return (viewport_x, viewport_y, int(viewport_w), int(viewport_h)), zoom_factor


def draw_results(image: np.ndarray, box_coords: tuple, zoom_val: float, color: tuple, label: str):
    """Helper function to draw the box and text on an image."""
    x, y, w, h = box_coords
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 4) # Draw the box
    
    # Prepare text
    text = f"{label} Zoom: {zoom_val:.2f}x"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Get text size to position it nicely inside the top-right corner
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x + w - text_w - 10 # 10px padding from the right edge
    text_y = y + text_h + 10     # 10px padding from the top edge
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
if __name__ == "__main__":

    INPUT_VIDEO = 'input.mp4'
    OUTPUT_FOLDER = 'output'
    MARGIN_VALUE = 50

    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    else:
        print(f"Cleaning output folder: '{OUTPUT_FOLDER}'...")
        shutil.rmtree(OUTPUT_FOLDER)
        os.makedirs(OUTPUT_FOLDER)

    print(f"Loading first frame from '{INPUT_VIDEO}'...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    ret, source_image = cap.read()
    cap.release()

    if ret:
        print("Successfully loaded frame. Starting analysis...")

        # --- Test 1: Raw Content Viewport (margin = 0) ---
        print("\nStep 1: Calculating viewport for raw content (margin=0)...")
        raw_viewport, raw_zoom = analyze_frame_for_zoom(source_image, margin=0)
        image_with_raw_box = source_image.copy()
        draw_results(image_with_raw_box, raw_viewport, raw_zoom, color=(0, 0, 255), label="Raw")
        output_path_raw = os.path.join(OUTPUT_FOLDER, 'detection_viewport_raw.png')
        cv2.imwrite(output_path_raw, image_with_raw_box)
        print(f"Saved raw viewport image to: '{output_path_raw}'")

        # --- Test 2: Final Viewport with Margin ---
        print(f"\nStep 2: Calculating viewport with a {MARGIN_VALUE}px margin...")
        margin_viewport, margin_zoom = analyze_frame_for_zoom(source_image, margin=MARGIN_VALUE)
        image_with_margin_box = source_image.copy()
        draw_results(image_with_margin_box, margin_viewport, margin_zoom, color=(0, 255, 0), label="Margin")
        output_path_margin = os.path.join(OUTPUT_FOLDER, 'detection_viewport_with_margin.png')
        cv2.imwrite(output_path_margin, image_with_margin_box)
        print(f"Saved margin viewport image to: '{output_path_margin}'")

        print("\nDone. Check the 'output' folder for the result images.")
    else:
        print(f"FATAL ERROR: Could not read first frame from '{INPUT_VIDEO}'.")