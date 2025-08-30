import csv
import cv2
import os
import time
import h5py
import numpy as np
from datetime import datetime
from multiprocessing import Process, Queue, Event  
from queue import Empty as QueueEmpty

def csv_writer(queue, csv_path):
    print(f"[Writer] Starting CSV writer at {csv_path}")
    header = [
        "cnt", "dr_cam_x", "dr_cam_y", "dr_cam_z", "err",
        "dr_lab_x", "dr_lab_y", "dr_lab_z",
        "r_cam_x", "r_cam_y", "r_cam_z",
        "r_lab_x", "r_lab_y", "r_lab_z",
        "posx", "posy", "heading", "step_dir", "step_mag",
        "intx", "inty", "ts", "seq"
    ]
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while True:
            row = queue.get()
            if row == "STOP":
                break
            writer.writerow(row)
    print("[Writer] CSV writing complete and file closed.")


import os
import time
import h5py
from queue import Empty as QueueEmpty


def noise_pattern_writer(queue, base_path="noise_patterns", run_tag="test", debug=False, flush_interval=100):
    """
    Continuously saves noise pattern frames from a queue into an HDF5 file.
    Exits only after receiving the 'STOP' sentinel so late frames aren't lost.
    
    Args:
        queue: A Queue providing frame data as dicts with keys:
               'timestamp', 'frame_number', 'original_image', 'bowl_image'.
        base_path (str): Directory where the .h5 file will be created.
        run_tag (str): Tag used in the filename.
        debug (bool): If True, print progress logs.
        flush_interval (int): How often (in frames) to flush data to disk.
    """

    # Small startup delay to let the producer warm up
    time.sleep(0.1)

    # Prepare file path and ensure output directory exists
    os.makedirs(base_path, exist_ok=True)
    filepath = os.path.join(base_path, f"noise_patterns_{run_tag}.h5")
    print(f"[NoiseWriter] Writing noise patterns to {filepath}")

    frame_count = 0

    # Open HDF5 file for writing
    with h5py.File(filepath, 'w') as f:
        # Groups for images
        original_images = f.create_group('original_images')
        bowl_images = f.create_group('bowl_images')

        # Datasets for metadata (timestamps, frame numbers)
        timestamps = f.create_dataset('timestamps', (0,), maxshape=(None,), dtype='int64')
        frame_numbers = f.create_dataset('frame_numbers', (0,), maxshape=(None,), dtype='int32')

        while True:
            try:
                # Wait up to 1s for next frame
                data = queue.get(timeout=1.0)
            except QueueEmpty:
                # No new frame yet → keep waiting
                continue

            # Stop condition
            if data == "STOP":
                if debug:
                    print(f"[NoiseWriter] Received STOP after {frame_count} frames")
                f.flush()  # Final flush
                break

            # Extract frame metadata
            ts = data['timestamp']
            num = data['frame_number']
            orig = data['original_image']
            bowl = data['bowl_image']

            # Expand metadata datasets by 1 and save values
            i = timestamps.shape[0]
            timestamps.resize((i + 1,))
            frame_numbers.resize((i + 1,))
            timestamps[i], frame_numbers[i] = ts, num

            # Save images under unique names
            frame_name = f'frame_{num:06d}'
            try:
                original_images.create_dataset(frame_name, data=orig, compression='gzip', compression_opts=1)
                bowl_images.create_dataset(frame_name, data=bowl, compression='gzip', compression_opts=1)
            except ValueError as e:
                if "name already exists" in str(e):
                    if debug:
                        print(f"[NoiseWriter] Warning: Frame {num} already exists, skipping")
                    continue
                raise  # Unexpected error → propagate

            # Increment frame count
            frame_count += 1

            # Always flush every N frames for safety
            if frame_count % flush_interval == 0:
                f.flush()
                if debug:
                    print(f"[NoiseWriter] Flushed after {frame_count} frames")

    print(f"[NoiseWriter] Finished saving {frame_count} frames to {filepath}")

