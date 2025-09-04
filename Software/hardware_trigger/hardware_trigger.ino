// Arduino: Serial Trigger + Handshake Receiver
// Listens for serial input and generates a TTL pulse on pin 8
// - 'T' → generate a TTL pulse on pin 8 and reply "RCVD"
// - 'H' → reply "HANDSHAKE_OK"

const int triggerPin = 8;        // Pin used for TTL output
const int handshakePin = 7;
const int pulseDuration = 10;    // pulse width in ms

void setup() {
  pinMode(triggerPin, OUTPUT);
  pinMode(handshakePin,OUTPUT);
  digitalWrite(triggerPin, LOW);
  digitalWrite(handshakePin, LOW);
  Serial.begin(9600); // Match Python baud rate
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'T') {  // Trigger command
      Serial.print("RCVD");
      digitalWrite(triggerPin, HIGH);
      delay(pulseDuration);
      digitalWrite(triggerPin, LOW);
    } 
    else if (cmd == 'H') {  // Handshake command
      Serial.print("HANDSHAKE_OK");
      digitalWrite(handshakePin, HIGH);
      delay(pulseDuration);
      digitalWrite(handshakePin, LOW);
    }
  }
}
