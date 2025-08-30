// Arduino: Serial Trigger Receiver
// Listens for serial input and generates a TTL pulse on pin 8

const int triggerPin = 8;   // Pin used for TTL output
const int pulseDuration = 10; // pulse width in ms

void setup() {
  pinMode(triggerPin, OUTPUT);
  digitalWrite(triggerPin, LOW);
  Serial.begin(9600); // Match Python baud rate
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'T') { // trigger command
      Serial.print("RCVD");
      digitalWrite(triggerPin, HIGH);
      delay(pulseDuration);
      digitalWrite(triggerPin, LOW);
    }
  }
}
