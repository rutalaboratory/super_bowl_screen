from utils import * 
from fictrac_client import *

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Safe for macOS/Windows

    queue = mp.Queue()
    csv_path = "fictrac_output.csv"

    writer_proc = mp.Process(target=csv_writer, args=(queue, csv_path))
    writer_proc.start()

    client = FicTracClient(queue=queue, debug=True)
    try:
        client.run()
    finally:
        queue.put("STOP")
        writer_proc.join()
