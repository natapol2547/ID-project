#include <Servo.h>

Servo myservo;  // Create servo object

const int inPin = 2;
const int servoPin = 8;
int pos = 95;
int turnAngle1 = 30;
int turnAngle2 = 50;
bool lastState = LOW;
int stage = 1;

void setup() {
  Serial.begin(9600);
  myservo.attach(servoPin);  // Attach servo
  pinMode(inPin, INPUT_PULLUP);     // Set pin as input
  myservo.write(pos);        // Set initial servo position
}

void loop() {
  bool inputState = digitalRead(inPin);
  // Serial.print(inputState);
  // Serial.print(" ");
  // Serial.println(stage);

  if (inputState && !lastState) {
    if (stage < 5){
      stage++;
    } else {
      stage = 1;
    }
  }
  
  lastState = inputState;

  switch (stage) {
    case 1:
      myservo.write(pos - turnAngle2);
      break;
    case 2:
      myservo.write(pos - turnAngle1);
      break;
    case 3:
      myservo.write(pos);
      break;
    case 4:
      myservo.write(pos + turnAngle1);
      break;
    case 5:
      myservo.write(pos + turnAngle2);
      break;
  }


}
