import socket
import time


class FicTracClient:
    """
    A simple FicTrac TCP client that reads one complete frame at a time.
    Each call to read_frame() returns a full data dictionary or None if disconnected.
    """

    def __init__(self, host='127.0.0.1', port=3000):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""

    def connect(self):
        """Connect to FicTrac TCP server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"[Client] Connected to FicTrac at {self.host}:{self.port}")


    def read_frame(self):
        """
        Read data from FicTrac socket until one full line (frame) is received.
        Returns a parsed dictionary with all FicTrac variables, or None if no data.
        """
        while True:
            # Read from socket
            new_data = self.sock.recv(1024)
            if not new_data:
                print("[Client] Connection closed by server.")
                self.close()
                return None

            # Append decoded data to buffer
            self.buffer += new_data.decode("UTF-8")

            # Split buffer into lines (frames)
            lines = self.buffer.split("\n")
            # If no complete frame, keep waiting
            if len(lines) < 2:
                continue

            # Process all complete frames, keep only the last valid one
            latest_data = None
            print(f"[Client] Buffer has {len(lines)} lines")
            for line in lines[:-1]:
                toks = line.split(", ")
                if (len(toks) < 24) or (toks[0] != "FT"):
                    print("[Client] Bad read:", line)
                    continue
                latest_data = {
                    "cnt": int(toks[1]),
                    "dr_cam": [float(toks[2]), float(toks[3]), float(toks[4])],
                    "err": float(toks[5]),
                    "dr_lab": [float(toks[6]), float(toks[7]), float(toks[8])],
                    "r_cam": [float(toks[9]), float(toks[10]), float(toks[11])],
                    "r_lab": [float(toks[12]), float(toks[13]), float(toks[14])],
                    "posx": float(toks[15]),
                    "posy": float(toks[16]),
                    "heading": float(toks[17]),
                    "step_dir": float(toks[18]),
                    "step_mag": float(toks[19]),
                    "intx": float(toks[20]),
                    "inty": float(toks[21]),
                    "ts": float(toks[22]),
                    "seq": int(toks[23]),
                    "python_ts": time.perf_counter_ns() / 1e9,
                }

            # Clear buffer up to last incomplete frame
            self.buffer = lines[-1]

            if latest_data:
                print(f"[Client] Read frame {latest_data['cnt']}")
                return latest_data

    def close(self):
        """Close the TCP connection."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            print("[Client] Connection closed.")
