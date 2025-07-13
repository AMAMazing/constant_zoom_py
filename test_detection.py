import cv2
import numpy as np
import os
import shutil

def find_content_bounding_box(frame: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    """
    Analyzes a frame to find a single bounding box that encloses all non-background content.
    This version uses a fixed threshold, which is more reliable for high-contrast images.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply a fixed binary threshold.
    # Any pixel with a grayscale value > 40 will be set to 255 (white), all others to 0 (black).
    # This reliably isolates the light text/emoji from the dark background.
    # This replaces the less reliable Otsu's method.
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # 3. Find contours of all separate content "blobs" on the thresholded image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = frame.shape[:2]
        print("Warning: No content detected. Defaulting to full frame.")
        return 0, 0, w, h

    # 4. Find the single master bounding box that encloses ALL individual contours
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)

    # 5. Apply the specified margin to the raw bounding box
    x_margin = max(0, x - margin)
    y_margin = max(0, y - margin)
    w_margin = min(frame.shape[1], (x + w + margin)) - x_margin
    h_margin = min(frame.shape[0], (y + h + margin)) - y_margin

    return x_margin, y_margin, w_margin, h_margin


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
if __name__ == "__main__":

    # --- Configuration ---
    INPUT_VIDEO = 'input.mp4'  # The video to analyze
    OUTPUT_FOLDER = 'output'
    MARGIN_VALUE = 50          # The margin in pixels to test with

    # --- Setup Output Folder ---
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    else:
        print(f"Cleaning output folder: '{OUTPUT_FOLDER}'...")
        shutil.rmtree(OUTPUT_FOLDER)
        os.makedirs(OUTPUT_FOLDER)

    # --- Load First Frame from Video ---
    print(f"Attempting to load first frame from '{INPUT_VIDEO}'...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    source_image = None
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open video file '{INPUT_VIDEO}'.")
    else:
        ret, source_image = cap.read()
        if not ret:
            print(f"FATAL ERROR: Could not read first frame from '{INPUT_VIDEO}'. The video might be empty or corrupt.")
        else:
            print(f"Successfully loaded first frame from '{INPUT_VIDEO}'.")
        cap.release()

    # --- Proceed with Analysis if Frame was Loaded ---
    if source_image is not None:
        # --- Test 1: Raw Bounding Box (margin = 0) ---
        print("\nStep 1: Calculating raw bounding box (margin=0)...")
        x_raw, y_raw, w_raw, h_raw = find_content_bounding_box(source_image, margin=0)
        image_with_raw_box = source_image.copy()
        cv2.rectangle(image_with_raw_box, (x_raw, y_raw), (x_raw + w_raw, y_raw + h_raw), (0, 0, 255), 4) # Red box

        output_path_raw = os.path.join(OUTPUT_FOLDER, 'detection_raw_box.png')
        cv2.imwrite(output_path_raw, image_with_raw_box)
        print(f"Saved image with raw detection box to: '{output_path_raw}'")

        # --- Test 2: Bounding Box with Margin ---
        print(f"\nStep 2: Calculating bounding box with a {MARGIN_VALUE}px margin...")
        x_m, y_m, w_m, h_m = find_content_bounding_box(source_image, margin=MARGIN_VALUE)
        image_with_margin_box = source_image.copy()
        cv2.rectangle(image_with_margin_box, (x_m, y_m), (x_m + w_m, y_m + h_m), (0, 255, 0), 4) # Green box

        output_path_margin = os.path.join(OUTPUT_FOLDER, 'detection_with_margin.png')
        cv2.imwrite(output_path_margin, image_with_margin_box)
        print(f"Saved image with margin box to: '{output_path_margin}'")

        print("\nDone. Check the 'output' folder for the result images.")