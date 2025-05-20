import Jetson.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)

GPIO.setup(24, GPIO.IN)

try:
    while True:
        print(GPIO.input(24))
except:
    GPIO.cleanup()