import socket
import time
import numpy as np
import subprocess
import math

class FicTracClient:
    """
    A client for connecting to the FicTrac socket server, reading and parsing motion data,
    and optionally sending parsed data rows to a multiprocessing queue.
    """

    def __init__(self, host='127.0.0.1', port=3000, queue=None, debug=False, stop_event=None):
        # Initialize connection parameters and internal variables
        self.host = host
        self.port = port
        self.sock = None  # Will hold the socket connection
        self.buffer = ""  # Buffer for accumulating incoming data
        self.queue = queue  # Optional multiprocessing queue to send parsed data
        self.debug = debug  # Enable timing/debug prints
        self.loop_times = []  # Store loop durations for performance diagnostics
        self.stop_event = stop_event  # Event flag for graceful shutdown

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.setblocking(0)
            print(f"[Client] Connected to FicTrac at {self.host}:{self.port}")

            # Log the connection event with NaNs for all FicTrac fields and timestamp only
            nan_row = [math.nan] * 24  # 24 fields expected from FicTrac
            python_ts = time.time()
            data_row = nan_row + [python_ts, "CONNECTED"]
            if self.queue:
                self.queue.put(data_row)
            return True
        except ConnectionRefusedError:
            print(f"[Client] Error: Could not connect to FicTrac at {self.host}:{self.port}")
            print("[Client] Please ensure that:")
            print("1. FicTrac is running")
            print("2. The correct host and port are specified in config.yaml")
            print("3. There are no firewall restrictions blocking the connection")
            if self.stop_event:
                self.stop_event.set()  # Signal to stop the main process
            return False
        except Exception as e:
            print(f"[Client] Unexpected error while connecting: {str(e)}")
            if self.stop_event:
                self.stop_event.set()  # Signal to stop the main process
            return False

    def read_data(self):
        """
        Read data from the socket, parse each line if it is a valid FicTrac output,
        and optionally send it to a multiprocessing queue.
        """
        try:
            while not self.stop_event.is_set():
                if self.debug:
                    start_time = time.time()

                # Receive data from the socket (blocking call)
                new_data = self.sock.recv(1024)
                if not new_data:
                    break  # Connection closed
                
                # Timestamp the exact moment the data is received
                python_ts = time.time()

                # Accumulate received data into buffer
                self.buffer += new_data.decode('UTF-8')

                while True:
                    endline = self.buffer.find("\n")  # Find end of one complete line
                    if endline == -1:
                        break  # No full line yet

                    # Extract and remove the line from the buffer
                    line = self.buffer[:endline]
                    self.buffer = self.buffer[endline + 1:]

                    # Split the line into comma-separated tokens
                    toks = line.split(", ")

                    # Ensure it's a valid FicTrac line with sufficient tokens
                    if (len(toks) < 24) or (toks[0] != "FT"):
                        print('[Client] Bad read')
                        continue

                    # Parse relevant fields
                    cnt = int(toks[1])
                    dr_cam = [float(toks[2]), float(toks[3]), float(toks[4])]
                    err = float(toks[5])
                    dr_lab = [float(toks[6]), float(toks[7]), float(toks[8])]
                    r_cam = [float(toks[9]), float(toks[10]), float(toks[11])]
                    r_lab = [float(toks[12]), float(toks[13]), float(toks[14])]
                    posx = float(toks[15])
                    posy = float(toks[16])
                    heading = float(toks[17])
                    step_dir = float(toks[18])
                    step_mag = float(toks[19])
                    intx = float(toks[20])
                    inty = float(toks[21])
                    ts = float(toks[22])
                    seq = int(toks[23])

                    # Final row includes FicTrac fields + Python timestamp + status field
                    data_row = [
                        cnt, *dr_cam, err, *dr_lab, *r_cam, *r_lab,
                        posx, posy, heading, step_dir, step_mag,
                        intx, inty, ts, seq,
                        python_ts, "DATA"
                    ]

                    # Send parsed data to queue if provided
                    if self.queue:
                        self.queue.put(data_row)

                    # Record loop timing if debugging
                    if self.debug:
                        end_time = time.time()
                        self.loop_times.append(end_time - start_time)

        finally:
            # Always close the socket on exit
            self.sock.close()
            # Optional: gracefully terminate FicTrac (disabled by default)
            # self.kill_fictrac_gracefully()
            print("[Client] Connection closed.")

            # Print performance stats if debugging
            if self.debug and self.loop_times:
                times = np.array(self.loop_times)
                avg = np.mean(times)
                std = np.std(times)
                print(f"[Timing] Loops: {len(times)} | Avg: {avg:.6f} s | Std: {std:.6f} s")

    def run(self):
        """
        Entry point to start the client: connect and start reading data.
        """
        if self.connect():
            self.read_data()
        else:
            print("[Client] Exiting due to connection failure")
