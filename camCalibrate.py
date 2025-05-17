# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2 or HQ Camera with wide-angle lens)
# connected to a NVIDIA Jetson Nano Developer Kit using OpenCV
import time
import cv2
import traceback
from calibration.camera_calibration_read import load_calibration_data


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3280,
    capture_height=2464,
    display_width=1640,
    display_height=1232,
    framerate=10,           # You can try higher FPS if needed
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class Camera():
    def __init__(self,width=1640, height=1232):
        self.window_title = "Wide FOV CSI Camera"
        window_handle = cv2.namedWindow(self.window_title, cv2.WINDOW_AUTOSIZE)
        mtx, dist = load_calibration_data()
        
        if mtx is None or dist is None:
            assert ValueError("Invalid calibration data")
        
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width,height), cv2.CV_32FC1)
        self.video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
            print("Released Video")
        cv2.destroyAllWindows()

    def capture(self):
        if not self.video_capture.isOpened():
            return "Error: Unable to open camera"

        ret_val, frame = self.video_capture.read()
        if not ret_val:
            print("Error: Could not read frame.")
        if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
            cv2.imshow(self.window_title, frame)
        # h,  w = frame.shape[:2]
        
        # Undistort the image
        frame = cv2.remap(frame, self._mapx, self._mapy, cv2.INTER_LINEAR)

        return frame


if __name__ == "__main__":
    print(gstreamer_pipeline())
    cam = Camera()
    # Capture an image
    image = cam.capture()
    # Save the image to a file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")
    cv2.imshow(cam.window_title, image)