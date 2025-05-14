import Jetson.GPIO as GPIO
import time
import cv2
import traceback

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=5000,     
    capture_height=5000,
    display_width=1200,
    display_height=1200,
    framerate=1,           # You can try higher FPS if needed
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

# Pin Definitions:
led_pin = 12 
cam_pin = 16
but_pin = 18  

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(led_pin, GPIO.OUT)
    GPIO.setup(but_pin, GPIO.IN)
    GPIO.setup(cam_pin,GPIO.OUT)
   
    GPIO.output(cam_pin,GPIO.LOW)
    GPIO.output(led_pin,GPIO.HIGH)
    print("Starting demo now! Press CTRL+C to exit")
    GPIO.add_event_detect(but_pin, GPIO.FALLING, bouncetime=200)

    try: 
        while True:
            if GPIO.event_detected(but_pin):
                GPIO.output(led_pin, GPIO.LOW)
                GPIO.output(cam_pin,GPIO.HIGH)
                # GPIO.output(cam_pin, GPIO.HIGH)
                print('nig')
                # while True:
                #     if GPIO.event_detected(but_pin):
                #         print("Button pressed!")
                #         GPIO.output(led_pin, GPIO.HIGH)
                #         time.sleep(1)
                #         GPIO.output(led_pin, GPIO.LOW)
                    
                #     time.sleep(0.1)
                time.sleep(0.1)
                GPIO.output(cam_pin, GPIO.LOW)
                time.sleep(3)
                GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting gracefully.")
    finally:
        GPIO.remove_event_detect(but_pin)  # <--- Important
        GPIO.cleanup()

if __name__ == '__main__':
    main()
