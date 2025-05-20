'''
Driving code for the game on jetson nano, includes main loop and GPIO interface
'''
import cv2
import random
import Jetson.GPIO as GPIO
import numpy as np
import time
from gameWindow import GameWindow
from gamePathLogic import checkTilePlacement, checkAnswerCorrectBool, imageToMatrix, questionDict, generate_grid_matrix_from_qr_images, findQuestionType
from camCalibrate import Camera
from calibration.perspective_transform_read import correct_perspective_images
from imageProcessing import split_image, process_image
from lightControl import Light
import os

CAPTURE_OUTPUT_DIR = "."
PERS_OUTPUT_DIR = "perspective_corrected_images"
SPLIT_OUTPUT_DIR = "split_images"

def take_images(camera:Camera):
    # start_time = time.time()
    # while True:
    #     elapsed = time.time() - start_time
    #     if elapsed > 2.5:
    #         break
    start_time = time.time()
    imagePaths = []
    for i in range(5):
        while True:
            frame = camera.capture()
            elapsed = time.time() - start_time
            if elapsed > 0.6:
                image_path = os.path.join(CAPTURE_OUTPUT_DIR, f"image_{i}.png")
                imagePaths.append(image_path)
                cv2.imwrite(image_path, frame)
                print(f"Saved image_{i}.png")
                GPIO.output(SERVO_PIN, GPIO.HIGH)
                GPIO.output(SERVO_PIN, GPIO.LOW)
                start_time = time.time()
                break
    return imagePaths

def main():
    #At start: 
    QUESTION_PIN = 24
    ANSWER_PIN = 18
    global SERVO_PIN
    SERVO_PIN = 16
    LED_PIN = 12 
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(QUESTION_PIN, GPIO.IN)
    GPIO.setup(ANSWER_PIN, GPIO.IN) 
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.add_event_detect(QUESTION_PIN, GPIO.RISING, bouncetime=200)
    GPIO.add_event_detect(ANSWER_PIN, GPIO.RISING, bouncetime=200)

    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(SERVO_PIN,GPIO.LOW)
    light = Light()
    light.on()
    #Display main start screen code here (Artboard 1)
    
    camera = Camera(debug=True)
    gameWindow = GameWindow("GameWIndow", debug=True)
    gameWindow.playSound(GameWindow.soundEffects.START)
    capturing_images = False
    randomStage = -1
    
    while True:
        camera.capture(correct_distortion=False)
        # question button pressed, randomize question from question dict
        if randomStage == -1 or GPIO.event_detected(QUESTION_PIN):
            print('change question')
            randomNumber = random.randint(1,29)
            gameWindow.displayStage(randomNumber, True)
            gameWindow.questionSoundEffect = findQuestionType(randomNumber)
            randomStage  = questionDict[randomNumber]
            # send question to display screen accordingly, note: question 1 = Artboard 2 .... question 29 = Artboard 30

        if GPIO.event_detected(ANSWER_PIN) and not capturing_images and randomStage != -1:
            capturing_images = True
            print('check answer')
            imagePaths = take_images(camera) #take images while rotating the servo
            capturing_images = False

            perspective_corrected_images = correct_perspective_images(imagePaths, output_dir=PERS_OUTPUT_DIR)
            # splitted_images = split_image(perspective_corrected_images, output_dir=SPLIT_OUTPUT_DIR, rows=1, cols=5, overlap_percent=15)
            # splitted_images = [process_image(image) for image in splitted_images]
            # stageMatrix = imageToMatrix(splitted_images)
            processed_image = []
            for idx, image in enumerate(perspective_corrected_images):
                processed_image.append(process_image(image, idx == 2))
            stageMatrix = generate_grid_matrix_from_qr_images(processed_image)
            print(stageMatrix)

            placementIsCorrect = checkTilePlacement(randomStage, stageMatrix)
            if placementIsCorrect:
                answerIsCorrect = checkAnswerCorrectBool(randomStage, stageMatrix)
                if answerIsCorrect:
                    image = np.zeros((300,300,3), dtype="uint8")
                    image[:] = (0, 255, 0)
                    cv2.imshow("Placement Image", image)
                    cv2.waitKey(2000)
                    gameWindow.playSound(gameWindow.questionSoundEffect)
                else:
                    image = np.zeros((300,300,3), dtype="uint8")
                    image[:] = (0, 255, 255)
                    cv2.imshow("Placement Image", image)
                    cv2.waitKey(2000)
                print(answerIsCorrect)
            else:
                image = np.zeros((300,300,3), dtype="uint8")
                image[:] = (0, 0, 255)
                cv2.imshow("Placement Image", image)
                cv2.waitKey(2000)
                print("Placement is not correct")
            cv2.destroyWindow("Placement Image")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        GPIO.cleanup()
