import Jetson.GPIO as GPIO
LED_PIN = 12

class Light:
    def __init__(self, pin: int = LED_PIN) -> None:
        self.pin = pin
        GPIO.setup(self.pin, GPIO.OUT)
    
    def on(self):
        GPIO.output(self.pin, GPIO.LOW)
    
    def off(self):
        GPIO.output(self.pin, GPIO.HIGH)

    def __del__(self):
        print("Turning Light off")
        self.off()