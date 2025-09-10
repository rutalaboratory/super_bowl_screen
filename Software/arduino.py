import serial
import numpy as np
import time

class ArduinoCommunication:

    def __init__(self,
                 port='/dev/ttyACM0',
                 baud_rate=115200,
                 handshake_interval=10):
        
        self.port = port
        self.baud_rate = baud_rate
        self.handshake_interval = handshake_interval  # seconds

        self.ser = serial.Serial(port, baud_rate)
        time.sleep(2)  # wait for the serial connection to initialize
        print("Arduino Initialized Successfully!")


    def send_trigger(self):
            self.ser.write(b'T')  # send trigger command
            trigger_time = time.perf_counter_ns()  # Timestamp in nanoseconds
            print("[Arduino] Trigger sent.")

            return trigger_time 
    
    def handshake(self):
            self.ser.write(b'H')  # send handshake command
            self.ser.flush()
        #     print("[Arduino] Handshake sent.")
