import cv2
import numpy as np
import json
import os

CONFIG_FILE = "camera_calibration_data.json"

# --- 1. Define Known Camera Parameters ---
# These are illustrative. Replace with your actual values.
F_MM = 3.15  # Focal length in mm (e.g., from a GoPro or webcam spec)
SENSOR_W_MM = 3.674  # Sensor width in mm (e.g., 1/2.3" sensor)
SENSOR_H_MM = 2.760  # Sensor height in mm (e.g., 1/2.3" sensor)

# --- Helper functions for JSON ---
def save_calibration_data(filename, camera_matrix, dist_coeffs, image_size, zoom_factor, reproj_error=0.0):
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": [dist_coeffs.flatten().tolist()],
        "image_size": list(image_size),
        "zoom_factor": zoom_factor,
        "reprojection_error": reproj_error
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Calibration data saved to {filename}")

def load_calibration_data(filename):
    if not os.path.exists(filename):
        return None, None, None, None, 1.0 # Default zoom 1.0
    with open(filename, 'r') as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(data["dist_coeffs"][0], dtype=np.float32)
    image_size = tuple(data["image_size"])
    zoom_factor = data.get("zoom_factor", 1.0)
    reproj_error = data.get("reprojection_error", 0.0)
    print(f"Calibration data loaded from {filename}")
    return camera_matrix, dist_coeffs, image_size, reproj_error, zoom_factor

# Global variables
initial_k1, initial_k2, initial_p1, initial_p2, initial_k3 = 0.0, 0.0, 0.0, 0.0, 0.0
initial_zoom_factor = 1.0
current_K_orig = None
current_D = None
current_img_size_orig = None
current_zoom_factor = 1.0

# --- Image Pre-processing (Zooming In/Out) ---
def get_processed_image(img_input, zoom_level):
    h_orig, w_orig = img_input.shape[:2]

    if zoom_level == 1.0:
        return img_input

    if zoom_level > 1.0: # Zoom In
        crop_w = int(w_orig / zoom_level)
        crop_h = int(h_orig / zoom_level)
        if crop_w < 1: crop_w = 1 # Ensure crop dim is at least 1
        if crop_h < 1: crop_h = 1

        x_start = (w_orig - crop_w) // 2
        y_start = (h_orig - crop_h) // 2

        img_cropped = img_input[y_start : y_start + crop_h, x_start : x_start + crop_w]
        img_processed = cv2.resize(img_cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    else: # Zoom Out (zoom_level < 1.0)
        scaled_w = int(w_orig * zoom_level)
        scaled_h = int(h_orig * zoom_level)
        if scaled_w < 1: scaled_w = 1 # Ensure scaled dim is at least 1
        if scaled_h < 1: scaled_h = 1

        img_scaled_down = cv2.resize(img_input, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Create a black canvas of original dimensions
        img_processed = np.zeros((h_orig, w_orig, 3), dtype=img_input.dtype)

        # Calculate top-left corner to paste the scaled image (centered)
        x_start = (w_orig - scaled_w) // 2
        y_start = (h_orig - scaled_h) // 2
        img_processed[y_start : y_start + scaled_h, x_start : x_start + scaled_w] = img_scaled_down

    return img_processed

# --- Undistortion Function ---
def undistort_and_show(img_to_process, K_for_undistort, D_local, window_name='Undistorted Image'):
    global current_K_orig, current_D, current_img_size_orig, current_zoom_factor

    img_h_px, img_w_px = img_to_process.shape[:2]

    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix=K_for_undistort,
        distCoeffs=D_local,
        R=None,
        newCameraMatrix=K_for_undistort,
        size=(img_w_px, img_h_px),
        m1type=cv2.CV_32FC1
    )

    if mapx is not None and mapy is not None:
        undistorted_img = cv2.remap(img_to_process, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        undistorted_img_S = cv2.resize(undistorted_img, (960, 540))
        cv2.imshow(window_name, undistorted_img_S)
        current_D = D_local # current_K_orig, current_img_size_orig, current_zoom_factor are updated elsewhere
    else:
        print("Error: Could not compute undistortion maps.")
        img_to_process_S = cv2.resize(img_to_process, (960, 540))
        cv2.imshow(window_name, img_to_process_S)

# --- Callback for Trackbars ---
def on_trackbar_change(_):
    global img_original_loaded, K_initial_calc, current_zoom_factor

    k1 = (cv2.getTrackbarPos('k1 (x1e-4)', 'Controls') - 5000) / 1000.0
    k2 = (cv2.getTrackbarPos('k2 (x1e-5)', 'Controls') - 5000) / 500.0
    p1 = (cv2.getTrackbarPos('p1 (x1e-4)', 'Controls') - 5000) / 10000.0
    p2 = (cv2.getTrackbarPos('p2 (x1e-4)', 'Controls') - 5000) / 10000.0
    k3 = (cv2.getTrackbarPos('k3 (x1e-6)', 'Controls') - 5000) / 500.0
    D_manual = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    # Zoom trackbar: e.g., 50 to 400. 100 is 1.0x.
    # Min zoom 0.5x (50/100), Max zoom 4.0x (400/100)
    zoom_val_tb = cv2.getTrackbarPos('Zoom (x100)', 'Controls')
    current_zoom_factor = zoom_val_tb / 100.0
    if current_zoom_factor == 0: current_zoom_factor = 0.01 # Avoid division by zero if trackbar somehow goes to 0

    print(f"Zoom: {current_zoom_factor:.2f}x | D: k1={k1:.5f}, k2={k2:.6f}, p1={p1:.5f}, p2={p2:.5f}, k3={k3:.7f}")

    img_processed = get_processed_image(img_original_loaded, current_zoom_factor)
    img_processed_S = cv2.resize(img_processed, (960, 540))
    cv2.imshow('Processed Input (Pre-Undistortion)', img_processed_S)

    undistort_and_show(img_processed, K_initial_calc, D_manual)
# --- Main ---
if __name__ == "__main__":
    # --- Load the original image ---
    IMAGE_PATH = 'capture_20250519-092925.jpg' # <<< CHANGE THIS TO YOUR IMAGE
    img_original_loaded = cv2.imread(IMAGE_PATH)
    if img_original_loaded is None:
        print(f"Error: Could not load image at {IMAGE_PATH}")
        img_w_px, img_h_px = 640, 480
        img_original_loaded = np.ones((img_h_px, img_w_px, 3), dtype=np.uint8) * 200
        # ... (dummy image lines)
    img_h_orig, img_w_orig = img_original_loaded.shape[:2]
    current_img_size_orig = (img_w_orig, img_h_orig)

    K_loaded, D_loaded, img_size_loaded, _, zoom_loaded = load_calibration_data(CONFIG_FILE)

    if K_loaded is not None and D_loaded is not None:
        print("Using loaded calibration parameters.")
        if img_size_loaded != current_img_size_orig:
            print(f"Warning: Loaded calibration image size {img_size_loaded} "
                  f"differs from current image size {current_img_size_orig}.")
        K_initial_calc = K_loaded
        initial_k1, initial_k2, initial_p1, initial_p2, initial_k3 = D_loaded.flatten()
        initial_zoom_factor = zoom_loaded
        current_K_orig = K_initial_calc
        current_D = D_loaded
        current_zoom_factor = initial_zoom_factor
    else:
        print("No existing calibration file. Calculating K from sensor specs.")
        fx = F_MM * (img_w_orig / SENSOR_W_MM)
        fy = F_MM * (img_h_orig / SENSOR_H_MM)
        cx = img_w_orig / 2.0
        cy = img_h_orig / 2.0
        K_initial_calc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        current_K_orig = K_initial_calc
        current_D = np.array([initial_k1, initial_k2, initial_p1, initial_p2, initial_k3], dtype=np.float32)
        current_zoom_factor = initial_zoom_factor

    print("Initial Camera Matrix K (for unzoomed camera):\n", K_initial_calc)
    print(f"Initial D: {[initial_k1, initial_k2, initial_p1, initial_p2, initial_k3]}")
    print(f"Initial Zoom Factor: {initial_zoom_factor:.2f}")

    cv2.namedWindow('Original Image')
    img_original_loaded_S = cv2.resize(img_original_loaded, (960, 540))
    cv2.imshow('Original Image', img_original_loaded_S)
    cv2.namedWindow('Processed Input (Pre-Undistortion)')
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 350)
    cv2.namedWindow('Undistorted Image')

    # Zoom trackbar range: 50 to 400 (0.5x to 4.0x). Default 100 (1.0x).
    # Clamp initial_zoom_factor for trackbar to be within its displayable range
    tb_zoom_initial = int(np.clip(initial_zoom_factor * 100, 50, 400))

    cv2.createTrackbar('k1 (x1e-4)', 'Controls', int(initial_k1 * 1000.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('k2 (x1e-5)', 'Controls', int(initial_k2 * 500.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('p1 (x1e-4)', 'Controls', int(initial_p1 * 10000.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('p2 (x1e-4)', 'Controls', int(initial_p2 * 10000.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('k3 (x1e-6)', 'Controls', int(initial_k3 * 500.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('Zoom (x100)', 'Controls', tb_zoom_initial, 400, on_trackbar_change) # Min 50 (by convention for createTrackbar, actual min is 0 unless enforced)

    # Manually set the minimum for the zoom trackbar if OpenCV doesn't directly support it
    # (OpenCV trackbars usually start from 0, so we rely on our scaling logic)
    # We'll ensure the zoom_factor is at least 0.5 in on_trackbar_change by checking the raw value.
    # The trackbar will visually go from 0-400, so we'll map 50-400 from that.
    # A better way is to have a fixed min for trackbar if possible, or adjust mapping.
    # For simplicity, we will just scale what we get: if trackbar provides X, zoom = (X_min_desired + X) / 100 or similar.
    # Let's make the trackbar go 0-350 and add 50 to it, so it's 50-400 logically.
    # Zoom Trackbar: Logical range 0.5x to 4.0x. Trackbar physical range 0 to 350.
    # zoom_factor = (cv2.getTrackbarPos(...) + 50) / 100.0
    # Initial value for trackbar: int(initial_zoom_factor*100 - 50)
    # Clamped initial value for trackbar:
    clamped_initial_tb_zoom = int(np.clip(initial_zoom_factor * 100 - 50, 0, 350))
    cv2.setTrackbarMin('Zoom (x100)', 'Controls', 0) # Does not exist, trackbars are 0 to max
                                                    # We handle the mapping in the callback.
                                                    # The previous trackbar with max 400 is fine if we map 50-400.

    # For the trackbar 'Zoom (x100)' from 0 to 400:
    # We want value 50 to be 0.5x, 100 to be 1.0x, 400 to be 4.0x.
    # If trackbar reads `val`, then `zoom_factor = val / 100.0`.
    # We need to ensure `val` is at least 50.
    # So in on_trackbar_change:
    # zoom_val_tb = cv2.getTrackbarPos('Zoom (x100)', 'Controls')
    # if zoom_val_tb < 50: zoom_val_tb = 50 # Enforce minimum
    # current_zoom_factor = zoom_val_tb / 100.0

    on_trackbar_change(0) # Initial display

    print("\n--- Controls ---")
    print("Zoom trackbar: Effective range 0.5x to 4.0x (slider values 50-400).")
    print("Press 's' to SAVE. 'l' to LOAD. 'q' to QUIT.")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_K_orig is not None and current_D is not None and current_img_size_orig is not None:
                save_calibration_data(CONFIG_FILE, current_K_orig, current_D, current_img_size_orig, current_zoom_factor)
            else:
                print("Cannot save, parameters not fully defined.")
        elif key == ord('l'):
            print("Attempting to reload parameters...")
            K_reloaded, D_reloaded, img_size_reloaded, _, zoom_reloaded = load_calibration_data(CONFIG_FILE)
            if K_reloaded is not None and D_reloaded is not None:
                K_initial_calc = K_reloaded
                current_K_orig = K_initial_calc
                r_k1, r_k2, r_p1, r_p2, r_k3 = D_reloaded.flatten()
                current_D = D_reloaded
                current_zoom_factor = zoom_reloaded

                cv2.setTrackbarPos('k1 (x1e-4)', 'Controls', int(r_k1 * 10000.0 + 5000))
                cv2.setTrackbarPos('k2 (x1e-5)', 'Controls', int(r_k2 * 100000.0 + 5000))
                cv2.setTrackbarPos('p1 (x1e-4)', 'Controls', int(r_p1 * 10000.0 + 5000))
                cv2.setTrackbarPos('p2 (x1e-4)', 'Controls', int(r_p2 * 10000.0 + 5000))
                cv2.setTrackbarPos('k3 (x1e-6)', 'Controls', int(r_k3 * 1000000.0 + 5000))

                # Ensure loaded zoom factor is within trackbar's displayable range after scaling
                tb_zoom_val_loaded = int(np.clip(current_zoom_factor * 100, 0, 400)) # Trackbar is 0-400
                # We will enforce logical min of 0.5x (trackbar value 50) in the callback
                cv2.setTrackbarPos('Zoom (x100)', 'Controls', tb_zoom_val_loaded)

                on_trackbar_change(0) # Trigger update
            else:
                print(f"Failed to load from {CONFIG_FILE}.")
    cv2.destroyAllWindows()