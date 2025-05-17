import cv2
import json
import numpy as np
import os

CALIBRATION_FILE = "camera_calibration_data.json" # Updated to .json

def load_calibration_data(filename =CALIBRATION_FILE):
    """Loads camera calibration data from a JSON file."""
    if not os.path.exists(filename):
        print(f"Error: Calibration file '{filename}' not found.")
        return None, None, None, 1.0 # K, D, img_size, zoom
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        # dist_coeffs from JSON is [[k1,k2,p1,p2,k3]], OpenCV expects (5,) or (1,5) or (5,1)
        dist_coeffs = np.array(data["dist_coeffs"][0], dtype=np.float32)
        image_size_calibrated = tuple(data["image_size"]) # Size for which K & D were defined
        zoom_factor = data.get("zoom_factor", 1.0) # Default to 1.0 if not present
        print(f"Calibration data loaded from '{filename}':")
        print(f"  Camera Matrix (K):\n{camera_matrix}")
        print(f"  Distortion Coefficients (D): {dist_coeffs.flatten()}")
        print(f"  Calibrated Image Size: {image_size_calibrated}")
        print(f"  Zoom Factor: {zoom_factor:.2f}")
        return camera_matrix, dist_coeffs, image_size_calibrated, zoom_factor
    except Exception as e:
        print(f"Error loading or parsing calibration file '{filename}': {e}")
        return None, None, None, 1.0


def zoom_image(img_input, zoom_level):
    """Applies zoom-in or zoom-out pre-processing."""
    h_orig, w_orig = img_input.shape[:2]

    if zoom_level == 1.0:
        return img_input

    if zoom_level > 1.0: # Zoom In
        crop_w = int(w_orig / zoom_level)
        crop_h = int(h_orig / zoom_level)
        if crop_w < 1: crop_w = 1
        if crop_h < 1: crop_h = 1

        x_start = (w_orig - crop_w) // 2
        y_start = (h_orig - crop_h) // 2

        img_cropped = img_input[y_start : y_start + crop_h, x_start : x_start + crop_w]
        img_processed = cv2.resize(img_cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    else: # Zoom Out (zoom_level < 1.0)
        scaled_w = int(w_orig * zoom_level)
        scaled_h = int(h_orig * zoom_level)
        if scaled_w < 1: scaled_w = 1
        if scaled_h < 1: scaled_h = 1

        img_scaled_down = cv2.resize(img_input, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        img_processed = np.zeros((h_orig, w_orig, 3), dtype=img_input.dtype) # Black canvas

        x_start = (w_orig - scaled_w) // 2
        y_start = (h_orig - scaled_h) // 2
        img_processed[y_start : y_start + scaled_h, x_start : x_start + scaled_w] = img_scaled_down
    return img_processed

# def zoom_image(img_original, zoom_factor):
#     """Loads an image, applies zoom and undistortion, and optionally saves it."""

#     h_current, w_current = img_original.shape[:2]
#     current_img_size = (w_current, h_current)
#     # print(f"Applying pre-processing zoom: {zoom_factor:.2f}x")
#     return zoom_image(img_original, zoom_factor)


def apply_distortion_correction(img_original, K, D):
    h_current, w_current = img_original.shape[:2]
    # 2. Undistort the processed image
    # K and D are from the calibration (for the original full FoV of that calibrated size)
    # newCameraMatrix is also K, to maintain the scale of the undistortion output relative to K.
    # The size for remap is the size of the img_processed_for_undistortion.
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=D,
        R=None,
        newCameraMatrix=K, # Use original K to prevent undistortion step from scaling
        size=(w_current, h_current), # Size of the image being remapped
        m1type=cv2.CV_32FC1
    )

    if mapx is None or mapy is None:
        print("Error: Could not compute undistortion maps.")
        return

    return cv2.remap(img_original, mapx, mapy, interpolation=cv2.INTER_LINEAR)

if __name__ == "__main__":

    # Load calibration parameters
    K, D, calib_img_size, zoom = load_calibration_data()

    if K is not None and D is not None:
        image = cv2.imread("./calibration_images/capture_20250516-162008.jpg")
        image = zoom_image(image, zoom)
        h_current, w_current = image.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(
            cameraMatrix=K,
            distCoeffs=D,
            R=None,
            newCameraMatrix=K, # Use original K to prevent undistortion step from scaling
            size=(w_current, h_current), # Size of the image being remapped
            m1type=cv2.CV_32FC1
        )
        image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        if image is not None:
            cv2.imshow("Undistorted Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Image processing failed.")
    else:
        print("Exiting due to error loading calibration data.")
