import socket
import time
import multiprocessing as mp
import numpy as np


class FicTracClient:
    """
    A client for connecting to the FicTrac socket server, reading and parsing motion data,
    and optionally sending parsed data rows to a multiprocessing queue.
    """

    def __init__(self, host='127.0.0.1', port=3000, queue=None, debug=False):
        """
        Initialize the client.

        Args:
            host (str): IP address of the FicTrac server.
            port (int): Port on which FicTrac is serving data.
            queue (multiprocessing.Queue): Queue to send parsed data to.
            debug (bool): Whether to measure and print execution timing.
        """
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        self.queue = queue
        self.debug = debug
        self.loop_times = []  # Stores duration of each processing loop

    def connect(self):
        """
        Establish socket connection to the FicTrac server.
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"[Client] Connected to FicTrac at {self.host}:{self.port}")

    def read_data(self):
        """
        Continuously read from the socket, parse lines, and send structured data to the queue.
        Handles multiple lines per read without dropping any data.
        """
        try:
            while True:
                if self.debug:
                    start_time = time.time()

                # Read raw bytes from socket
                new_data = self.sock.recv(1024)
                if not new_data:
                    break  # FicTrac closed the connection

                # Decode and append to internal buffer
                self.buffer += new_data.decode('UTF-8')

                # === NEW: Process all complete lines ===
                while True:
                    endline = self.buffer.find("\n")
                    if endline == -1:
                        break  # Wait for more data to complete the line

                    line = self.buffer[:endline]
                    self.buffer = self.buffer[endline + 1:]

                    toks = line.split(", ")

                    if (len(toks) < 24) or (toks[0] != "FT"):
                        print('[Client] Bad read')
                        continue

                    # === Parse data ===
                    cnt      = int(toks[1])
                    dr_cam   = [float(toks[2]), float(toks[3]), float(toks[4])]
                    err      = float(toks[5])
                    dr_lab   = [float(toks[6]), float(toks[7]), float(toks[8])]
                    r_cam    = [float(toks[9]), float(toks[10]), float(toks[11])]
                    r_lab    = [float(toks[12]), float(toks[13]), float(toks[14])]
                    posx     = float(toks[15])
                    posy     = float(toks[16])
                    heading  = float(toks[17])
                    step_dir = float(toks[18])
                    step_mag = float(toks[19])
                    intx     = float(toks[20])
                    inty     = float(toks[21])
                    ts       = float(toks[22])
                    seq      = int(toks[23])

                    data_row = [
                        cnt, *dr_cam, err, *dr_lab, *r_cam, *r_lab,
                        posx, posy, heading, step_dir, step_mag,
                        intx, inty, ts, seq
                    ]

                    if self.queue:
                        self.queue.put(data_row)

                    if self.debug:
                        end_time = time.time()
                        self.loop_times.append(end_time - start_time)

        finally:
            self.sock.close()
            print("[Client] Connection closed.")

            if self.debug and self.loop_times:
                times = np.array(self.loop_times)
                avg = np.mean(times)
                std = np.std(times)
                print(f"[Timing] Loops: {len(times)} | Avg: {avg:.6f} s | Std: {std:.6f} s")


    def run(self):
        """
        Connect to the server and begin reading data.
        """
        self.connect()
        self.read_data()
