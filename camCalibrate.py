# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2 or HQ Camera with wide-angle lens)
# connected to a NVIDIA Jetson Nano Developer Kit using OpenCV
import time
import cv2
import traceback
from camera_calibration_read import load_calibration_data


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3280,
    capture_height=2464,
    display_width=1920,  # 102
    display_height=1080,  # 77
    framerate=20,           # You can try higher FPS if needed
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


def show_camera():
    window_title = "Wide FOV CSI Camera"

    mtx, dist = load_calibration_data()
    print("Using GStreamer pipeline:")

    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    print("Error: Could not read frame.")
                    break

                h,  w = frame.shape[:2]
                print(mtx,dist,(w,h))
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                    mtx, dist, (w, h), 1, (w, h))
                # undistort
                dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

                # crop the image
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, dst)
                else:
                    break

                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    cv2.imwrite("image1.jpg", frame)
                    # time.sleep(2)
                    # cv2.imwrite("image2.jpg", frame)
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
        print(gstreamer_pipeline())
        show_camera()
