import serial
import numpy as np
import time

class ArduinoCommunication:

    def __init__(self,
                 port='/dev/ttyACM0',
                 baud_rate=9600):
        
        self.port = port
        self.baud_rate = baud_rate

        self.ser = serial.Serial(port, baud_rate)
        print("Arduino Initialized Successfully!")


    def send_trigger(self):
            self.ser.write(b'T')  # send trigger command
            trigger_time = time.time()
            print("Trigger sent.")

            return trigger_time