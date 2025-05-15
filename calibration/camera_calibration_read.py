import cv2
import json
import numpy as np

CALIBRATION_FILE = "camera_calibration_data.json" # Updated to .json

def load_calibration_data(filepath=CALIBRATION_FILE):
    with open(filepath, 'r') as f: # Open in text mode for json
        data = json.load(f)
    # Convert lists back to numpy arrays
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])
    # image_size = data['image_size'] # This line can be uncommented if image_size is needed
    print("Calibration data loaded successfully.")
    return camera_matrix, dist_coeffs

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
