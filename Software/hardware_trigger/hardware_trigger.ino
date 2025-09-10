const int triggerPin = 8;
const int handshakePin = 7;

// Set your pulse duration here (in millisecond)
const int pulseDuration = 1;  // 1 ms pulse

void setup() {
  pinMode(triggerPin, OUTPUT);
  pinMode(handshakePin, OUTPUT);
  digitalWrite(triggerPin, LOW);
  digitalWrite(handshakePin, LOW);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'T') {
      digitalWrite(triggerPin, HIGH);
      delay(pulseDuration);
      digitalWrite(triggerPin, LOW);
    } 
    else if (cmd == 'H') {
      digitalWrite(handshakePin, HIGH);
      delay(pulseDuration);
      digitalWrite(handshakePin, LOW);
    }
  }
}
