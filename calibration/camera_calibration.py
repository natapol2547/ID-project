import numpy as np
import cv2
import glob
import os
import pickle # Alternative: np.savez for numpy arrays

# -----------------------------
# Configuration
# -----------------------------
# Number of internal corners of the checkerboard
# If your checkerboard is 7x5 SQUARES, then it's (7-1)x(5-1) = 6x4 internal corners.
# Adjust if your definition of "7x5" is different (e.g., 7x5 internal corners directly).
CHECKERBOARD_ROWS = 7  # Number of squares along the height - 1
CHECKERBOARD_COLS = 6  # Number of squares along the width - 1
INTERNAL_CORNERS_SHAPE = (CHECKERBOARD_COLS - 1, CHECKERBOARD_ROWS - 1) # (cols-1, rows-1) -> (6,4)

# Square size (in any consistent unit, e.g., mm, cm, m).
# This is only important if you want to measure real-world distances later.
# For distortion correction alone, it can be set to 1.
SQUARE_SIZE = 1.6  # e.g., 2.5 for 2.5 cm squares

IMAGE_DIR = "calibration_images"
SUPPORTED_FORMATS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
CALIBRATION_FILE = "camera_calibration_data.pkl" # Or .npz if using np.savez

# Display options
SHOW_DETECTED_CORNERS = True # Set to False to speed up if you have many images
# -----------------------------

def calibrate_camera():
    """
    Performs camera calibration using checkerboard images.
    Saves calibration data (camera matrix, distortion coefficients).
    """
    print(f"Looking for images in: {IMAGE_DIR}")
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Directory '{IMAGE_DIR}' not found. Please create it and add calibration images.")
        return

    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # These are the 3D coordinates of the checkerboard corners in an idealized world.
    objp = np.zeros((INTERNAL_CORNERS_SHAPE[0] * INTERNAL_CORNERS_SHAPE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:INTERNAL_CORNERS_SHAPE[0], 0:INTERNAL_CORNERS_SHAPE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE # Scale by square size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane.
    image_files = []

    for fmt in SUPPORTED_FORMATS:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, fmt)))

    if not image_files:
        print(f"No images found in '{IMAGE_DIR}'. Make sure images are present and have supported formats.")
        return

    print(f"Found {len(image_files)} images. Processing...")
    image_size = None # To be determined from the first image

    for i, fname in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {fname}")
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1] # (width, height)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, INTERNAL_CORNERS_SHAPE, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if SHOW_DETECTED_CORNERS:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, INTERNAL_CORNERS_SHAPE, corners2, ret)
                # Scale image for display if too large
                display_img = img.copy()
                h, w = display_img.shape[:2]
                max_disp_h = 720
                if h > max_disp_h:
                    scale_factor = max_disp_h / h
                    display_img = cv2.resize(display_img, (int(w * scale_factor), int(h * scale_factor)))

                cv2.imshow('Detected Corners', display_img)
                cv2.waitKey(0) # Wait indefinitely for a key press
        else:
            print(f"  Checkerboard corners not found in {fname}. Skipping.")

    if SHOW_DETECTED_CORNERS:
        cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("Error: No usable checkerboard patterns found in any images. Calibration failed.")
        print("Tips: Ensure good lighting, full checkerboard visibility, varied angles, and correct CHECKERBOARD_ROWS/COLS.")
        return

    print(f"\nCalibrating camera using {len(objpoints)} valid images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    if not ret:
        print("Camera calibration failed. Could not compute parameters.")
        return

    print("\nCalibration successful!")

    # Convert numpy arrays to lists for JSON serialization
    mtx_list = mtx.tolist() if mtx is not None else None
    dist_list = dist.tolist() if dist is not None else None

    # Save the camera calibration results
    calibration_data = {
        'camera_matrix': mtx_list,
        'dist_coeffs': dist_list,
        'image_size': image_size, # tuple is fine for JSON
        'reprojection_error': ret # float is fine for JSON
    }

    # Update file extension for JSON
    json_calibration_file = os.path.splitext(CALIBRATION_FILE)[0] + ".json"

    import json # Make sure to import json

    # Using json to save
    try:
        with open(json_calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=4) # indent for readability
        print(f"\nCalibration data saved to: {json_calibration_file}")
    except TypeError as e:
        print(f"Error saving to JSON: {e}")
        print("Make sure all data types are JSON serializable (e.g., lists instead of numpy arrays).")
    except Exception as e:
        print(f"An unexpected error occurred while saving to JSON: {e}")


if __name__ == '__main__':
    calibrate_camera()


