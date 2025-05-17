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
def save_calibration_data(filename, camera_matrix, dist_coeffs, image_size, reproj_error=0.0):
    """Saves camera calibration data to a JSON file."""
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": [dist_coeffs.flatten().tolist()], # Ensure it's a list containing a list
        "image_size": list(image_size),
        "reprojection_error": reproj_error # Placeholder for manual calibration
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Calibration data saved to {filename}")

def load_calibration_data(filename):
    """Loads camera calibration data from a JSON file."""
    if not os.path.exists(filename):
        return None, None, None, None
    with open(filename, 'r') as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    # dist_coeffs from JSON is [[k1,k2,p1,p2,k3]], OpenCV expects (5,) or (1,5) or (5,1)
    dist_coeffs = np.array(data["dist_coeffs"][0], dtype=np.float32)
    image_size = tuple(data["image_size"])
    reproj_error = data.get("reprojection_error", 0.0) # .get for backward compatibility
    print(f"Calibration data loaded from {filename}")
    return camera_matrix, dist_coeffs, image_size, reproj_error

# Global variables for trackbar values (to pre-fill if loading)
initial_k1, initial_k2, initial_p1, initial_p2, initial_k3 = 0.0, 0.0, 0.0, 0.0, 0.0
current_K = None
current_D = None
current_img_size = None

# --- Undistortion Function (called by trackbars or on load) ---
def undistort_and_show(img_local, K_local, D_local, window_name='Undistorted Image'):
    global current_K, current_D, current_img_size
    img_h_px, img_w_px = img_local.shape[:2]

    # Use original K as newCameraMatrix to prevent zooming/scaling changes
    # R=None as we are not doing stereo rectification
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix=K_local,
        distCoeffs=D_local,
        R=None,
        newCameraMatrix=K_local, # Key part to prevent zoom
        size=(img_w_px, img_h_px),
        m1type=cv2.CV_32FC1  # Type of the first output map
    )

    if mapx is not None and mapy is not None:
        undistorted_img = cv2.remap(img_local, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, undistorted_img)
        # Store the current K and D that produced this result
        current_K = K_local
        current_D = D_local
        current_img_size = (img_w_px, img_h_px)
    else:
        print("Error: Could not compute undistortion maps.")
        cv2.imshow(window_name, img_local) # Show original if map computation failed

# --- Callback for Trackbars ---
def on_trackbar_change(_): # The argument is the trackbar position, but we read all
    global img, K_initial_calc # Use the initially calculated K

    # Read all trackbar positions
    # Note: k values are scaled for finer control with integer trackbars
    k1 = (cv2.getTrackbarPos('k1 (x1e-4)', 'Controls') - 5000) / 5000.0  # Range e.g. -0.5 to +0.5
    k2 = (cv2.getTrackbarPos('k2 (x1e-5)', 'Controls') - 5000) / 10000.0 # Range e.g. -0.05 to +0.05
    p1 = (cv2.getTrackbarPos('p1 (x1e-4)', 'Controls') - 5000) / 10000.0
    p2 = (cv2.getTrackbarPos('p2 (x1e-4)', 'Controls') - 5000) / 10000.0
    k3 = (cv2.getTrackbarPos('k3 (x1e-6)', 'Controls') - 5000) / 1000000.0 # Range e.g. -0.005 to +0.005

    D_manual = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    print(f"Current D: k1={k1:.5f}, k2={k2:.6f}, p1={p1:.5f}, p2={p2:.5f}, k3={k3:.7f}")
    undistort_and_show(img, K_initial_calc, D_manual)


# --- Main ---
if __name__ == "__main__":
    # --- Load an image you want to correct ---
    IMAGE_PATH = './calibration_images/capture_20250516-162008.jpg' # <<< CHANGE THIS TO YOUR IMAGE
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load image at {IMAGE_PATH}")
        print("Creating a dummy image for demonstration.")
        img_w_px, img_h_px = 640, 480
        img = np.ones((img_h_px, img_w_px, 3), dtype=np.uint8) * 200
        cv2.line(img, (50, 50), (img_w_px - 50, 50), (0,0,0), 2)
        cv2.line(img, (50, img_h_px - 50), (img_w_px - 50, img_h_px - 50), (0,0,0), 2)
        cv2.line(img, (50, 50), (50, img_h_px - 50), (0,0,0), 2)
        cv2.line(img, (img_w_px - 50, 50), (img_w_px - 50, img_h_px - 50), (0,0,0), 2)
        cv2.line(img, (img_w_px // 2, 50), (img_w_px // 2, img_h_px - 50), (0,0,0), 2)
        cv2.line(img, (50, img_h_px // 2), (img_w_px - 50, img_h_px // 2), (0,0,0), 2)
    img_h_px, img_w_px = img.shape[:2]
    current_img_size = (img_w_px, img_h_px) # Store initial image size

    # --- Try to load existing calibration data ---
    K_loaded, D_loaded, img_size_loaded, _ = load_calibration_data(CONFIG_FILE)

    if K_loaded is not None and D_loaded is not None:
        print("Using loaded calibration parameters.")
        # Check if image size matches
        if img_size_loaded != (img_w_px, img_h_px):
            print(f"Warning: Loaded calibration image size {img_size_loaded} "
                  f"differs from current image size {(img_w_px, img_h_px)}.")
            print("Proceeding with loaded K, D but current image size for display.")
            # Potentially, one might want to scale K if image size changed significantly
            # but for simple undistortion, using original K might still be fine.
        K_initial_calc = K_loaded # Use loaded K
        initial_k1, initial_k2, initial_p1, initial_p2, initial_k3 = D_loaded.flatten()
        current_D = D_loaded # Store for potential immediate save if no tuning
        current_K = K_loaded
    else:
        print("No existing calibration file found or error loading. Calculating K from sensor specs.")
        # Calculate K from scratch if no config file
        fx = F_MM * (img_w_px / SENSOR_W_MM)
        fy = F_MM * (img_h_px / SENSOR_H_MM) # Assuming square pixels, fx and fy would be similar if sensor aspect ratio matches image
        cx = img_w_px / 2.0
        cy = img_h_px / 2.0
        K_initial_calc = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0, 0, 1]], dtype=np.float32)
        # initial_k values remain 0.0
        current_K = K_initial_calc # Store for potential immediate save
        current_D = np.array([initial_k1, initial_k2, initial_p1, initial_p2, initial_k3], dtype=np.float32)

    print("Initial Camera Matrix K (used for undistortion):\n", K_initial_calc)
    print(f"Initial Distortion Coefficients D: {[initial_k1, initial_k2, initial_p1, initial_p2, initial_k3]}")

    # --- Setup GUI ---
    cv2.namedWindow('Original Image')
    cv2.imshow('Original Image', img)
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL) # Normal so it can be resized
    cv2.resizeWindow('Controls', 400, 300)
    cv2.namedWindow('Undistorted Image')


    # Create trackbars. The ranges are scaled.
    # Default value: 5000 (center). Max value: 10000.
    # k1: (val - 5000) / 10000.0  => range -0.5 to 0.5
    # k2: (val - 5000) / 100000.0 => range -0.05 to 0.05
    # p1: (val - 5000) / 10000.0  => range -0.5 to 0.5
    # p2: (val - 5000) / 10000.0  => range -0.5 to 0.5
    # k3: (val - 5000) / 1000000.0=> range -0.005 to 0.005
    cv2.createTrackbar('k1 (x1e-4)', 'Controls', int(initial_k1 * 10000.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('k2 (x1e-5)', 'Controls', int(initial_k2 * 100000.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('p1 (x1e-4)', 'Controls', int(initial_p1 * 10000.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('p2 (x1e-4)', 'Controls', int(initial_p2 * 10000.0 + 5000), 10000, on_trackbar_change)
    cv2.createTrackbar('k3 (x1e-6)', 'Controls', int(initial_k3 * 1000000.0 + 5000), 10000, on_trackbar_change)

    # Initial undistortion display
    # If params were loaded, D_loaded will be used. Otherwise, initial (zero) D will be used.
    initial_D_for_display = np.array([initial_k1, initial_k2, initial_p1, initial_p2, initial_k3], dtype=np.float32)
    undistort_and_show(img, K_initial_calc, initial_D_for_display)

    print("\n--- Controls ---")
    print("Adjust sliders in the 'Controls' window to correct distortion.")
    print("Press 's' to SAVE current K and D parameters to JSON.")
    print("Press 'l' to LOAD parameters from JSON and re-apply (resets sliders).")
    print("Press 'q' to QUIT.")

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_K is not None and current_D is not None and current_img_size is not None:
                save_calibration_data(CONFIG_FILE, current_K, current_D, current_img_size)
            else:
                print("Cannot save, current parameters are not fully defined yet (undistort first).")
        elif key == ord('l'):
            print("Attempting to reload parameters...")
            K_reloaded, D_reloaded, img_size_reloaded, _ = load_calibration_data(CONFIG_FILE)
            if K_reloaded is not None and D_reloaded is not None:
                K_initial_calc = K_reloaded # Update the K being used
                r_k1, r_k2, r_p1, r_p2, r_k3 = D_reloaded.flatten()

                # Update trackbar positions
                cv2.setTrackbarPos('k1 (x1e-4)', 'Controls', int(r_k1 * 10000.0 + 5000))
                cv2.setTrackbarPos('k2 (x1e-5)', 'Controls', int(r_k2 * 100000.0 + 5000))
                cv2.setTrackbarPos('p1 (x1e-4)', 'Controls', int(r_p1 * 10000.0 + 5000))
                cv2.setTrackbarPos('p2 (x1e-4)', 'Controls', int(r_p2 * 10000.0 + 5000))
                cv2.setTrackbarPos('k3 (x1e-6)', 'Controls', int(r_k3 * 1000000.0 + 5000))
                # The setTrackbarPos will trigger on_trackbar_change, which calls undistort_and_show
                # Or, call it explicitly if setTrackbarPos doesn't always trigger:
                # on_trackbar_change(0) # or undistort_and_show(img, K_initial_calc, D_reloaded)
            else:
                print(f"Failed to load from {CONFIG_FILE}. Using current/default parameters.")


    cv2.destroyAllWindows()