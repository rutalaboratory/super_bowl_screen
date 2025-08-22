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

def noise_pattern_writer(queue, base_path="noise_patterns", debug=False):
    """
    Save noise patterns from a queue to HDF5 file.
    
    Args:
        queue: Multiprocessing queue containing noise pattern frames
        base_path: Base directory to save the patterns
        debug: Enable debug output (frame timing info, etc.)
    """
    # Add a small delay to let processes initialize
    time.sleep(0.1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Create HDF5 file
    filename = f"noise_patterns_{timestamp}.h5"
    filepath = os.path.join(base_path, filename)
    print(f"[NoiseWriter] Starting noise pattern writer at {filepath}")
    
    frame_count = 0
    with h5py.File(filepath, 'w') as f:
        # Create datasets to store frames and metadata
        # We'll create these with maxshape=(None,) to allow resizing
        frames_group = f.create_group('frames')
        timestamps_ds = f.create_dataset('timestamps', (0,), maxshape=(None,), dtype='float64')
        frame_numbers_ds = f.create_dataset('frame_numbers', (0,), maxshape=(None,), dtype='int32')
        
        while True:
            try:
                # Use timeout to prevent hanging if main process dies
                data = queue.get(timeout=5)
                
                if data == "STOP":
                    # Just exit, we've processed all frames already
                    break
                
                # Process normal frame
                frame_metadata = data
                timestamp = frame_metadata['timestamp']
                frame_number = frame_metadata['frame_number']
                frame = frame_metadata['frame_data']
                
                # Store the frame and metadata
                current_size = timestamps_ds.shape[0]
                new_size = current_size + 1
                timestamps_ds.resize((new_size,))
                frame_numbers_ds.resize((new_size,))
                
                frame_name = f'frame_{frame_number:06d}'  # Use actual frame number instead of count
                try:
                    frames_group.create_dataset(frame_name, data=frame, compression='gzip', compression_opts=1)
                    timestamps_ds[current_size] = timestamp
                    frame_numbers_ds[current_size] = frame_number
                except ValueError as e:
                    if "name already exists" in str(e):
                        print(f"[NoiseWriter] Warning: Frame {frame_number} already exists, skipping...")
                        continue
                    else:
                        raise e
                
                frame_count += 1
                
                if frame_count % 100 == 0 and debug:
                    print(f"[NoiseWriter] Saved {frame_count} frames")
                    f.flush()  # Periodically flush to disk
                    
            except QueueEmpty:
                print("[NoiseWriter] Timeout waiting for data, ensuring all frames are saved...")
                break
    
    print(f"[NoiseWriter] Completed saving {frame_count} frames to {filepath}")
