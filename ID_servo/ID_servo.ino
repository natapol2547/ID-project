#include <Servo.h>

Servo myservo;  // Create servo object

const int inPin = 2;
const int servoPin = 8;
int pos = 95;
int turnAngle1 = 30;
int turnAngle2 = 50;
bool lastState = LOW;
int stage = 0;

void setup() {
  Serial.begin(9600);
  myservo.attach(servoPin);  // Attach servo
  pinMode(inPin, INPUT_PULLUP);     // Set pin as input
  myservo.write(pos);        // Set initial servo position
}

void loop() {
  bool inputState = digitalRead(inPin);
  Serial.print(inputState);
  Serial.print(" ");
  Serial.println(stage);

  // if (inputState == HIGH && lastState == LOW) {
  //   myservo.write(pos + turnAngle2);
  //   delay(500);
  //   myservo.write(pos + turnAngle1);
  //   delay(500);
  //   myservo.write(pos);
  //   delay(500);
  //   myservo.write(pos - turnAngle1);
  //   delay(500);
  //   myservo.write(pos - turnAngle2);
  //   lastState = HIGH;
  // } else if (inputState == LOW && lastState == HIGH) {
  //   myservo.write(pos);
  //   lastState = LOW;
  // }

  switch (stage) {
    case 0:
      myservo.write(pos);
      if (inputState == HIGH) {
        Serial.println("nigga");
        stage = 1;
      }
      break;
    case 1:
      myservo.write(pos + turnAngle2);
      if (inputState == LOW) {
        stage = 2;
      }
      break;
    case 2:
      myservo.write(pos + turnAngle1);
      if (inputState == HIGH) {
        stage = 3;
      }
      break;
    case 3:
      myservo.write(pos);
      if (inputState == LOW) {
        stage = 4;
      }
      break;
    case 4:
      myservo.write(pos - turnAngle1);
      if (inputState == HIGH) {
        stage = 5;
      }
      break;
    case 5:
      myservo.write(pos - turnAngle2);
      if (inputState == LOW) {
        stage = 0;
      }
      break;
  }

  delay(500);
}
