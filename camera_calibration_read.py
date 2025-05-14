import cv2
import json
import numpy as np # if using .npz

CALIBRATION_FILE = "camera_calibration.json"

def load_calibration_data(filepath=CALIBRATION_FILE):
    try:
        with open(filepath, 'rb') as f:
            data = json.load(f)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        # image_size = data['image_size'] # if needed
        print("Calibration data loaded successfully.")
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print(f"Error: Calibration file '{filepath}' not found.")
        return None, None


# Example usage:
mtx, dist = load_calibration_data()
# if mtx is not None and dist is not None:
#     # Load an image
#     img = cv2.imread("some_image_from_same_camera.jpg")
#     if img is not None:
#         h, w = img.shape[:2]
#         new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#         undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
#         cv2.imshow("Undistorted Image", undistorted_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
