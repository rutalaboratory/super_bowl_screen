import h5py
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json 

def summarize_noise_file(filepath):
    """
    Print a summary of the noise_patterns HDF5 file.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with h5py.File(filepath, 'r') as f:
        # Access groups
        orig_group = f['original_images']
        bowl_group = f['bowl_images']
        timestamps = f['timestamps'][:]
        frame_numbers = f['frame_numbers'][:]

        print(f"Summary of file: {filepath}")
        print("-" * 40)
        print(f"Total frames saved: {len(frame_numbers)}")
        if len(frame_numbers) > 0:
            print(f"Frame numbers: {frame_numbers[0]} → {frame_numbers[-1]}")
            print(f"Timestamps: {timestamps[0]} → {timestamps[-1]}")
        print(f"Original images stored: {len(orig_group)}")
        print(f"Bowl images stored:     {len(bowl_group)}")
        print("Groups in file:", list(f.keys()))

def show_frame(filepath, frame_number):
    """
    Display the original and bowl images for a given frame number
    with timestamp overlay using OpenCV.
    """
    with h5py.File(filepath, 'r') as f:
        timestamps = f['timestamps'][:]
        frame_numbers = f['frame_numbers'][:]

        # Find index of the requested frame
        if frame_number not in frame_numbers:
            print(f"Frame {frame_number} not found in file.")
            return
        idx = np.where(frame_numbers == frame_number)[0][0]
        ts = timestamps[idx]

        frame_name = f'frame_{frame_number:06d}'

        print(frame_name)
        orig_img = f['original_images'][frame_name][:]
        bowl_img = f['bowl_images'][frame_name][:]

        # Overlay timestamp text on the images
        text = f"Frame {frame_number} | Timestamp: {ts:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_img, text, (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(bowl_img, text, (10, 30), font, 0.8, (0, 255, 0), 2)

        # Show the images
        cv2.imshow("Original Image", orig_img)
        cv2.imshow("Bowl Image", bowl_img)

        print(f"Displaying frame {frame_number} at timestamp {ts:.3f}.")
        print("Press any key in the image window to close.")

        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()

def find_duplicate_timestamp_frames(filepath):
    """
    Find frames where the difference in timestamps between consecutive frames is 0.
    
    Args:
        filepath (str): Path to the .h5 file
    
    Returns:
        duplicates (list of tuples): List of (frame_number_prev, frame_number_curr, timestamp)
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    
    with h5py.File(filepath, "r") as f:
        timestamps = f["timestamps"][:]
        frame_numbers = f["frame_numbers"][:]
    
    if len(timestamps) < 2:
        print("Not enough frames to check for duplicates.")
        return []
    
    # Find where timestamp difference is zero
    diffs = np.diff(timestamps)
    zero_idx = np.where(diffs == 0)[0]
    
    duplicates = []
    for idx in zero_idx:
        duplicates.append((int(frame_numbers[idx]), int(frame_numbers[idx+1]), float(timestamps[idx])))
    
    if duplicates:
        print("Frames with duplicate timestamps found:")
        for prev_f, curr_f, ts in duplicates:
            print(f"  Frame {prev_f} and Frame {curr_f} → Timestamp = {ts}")
    else:
        print("No duplicate timestamps found.")
    
    return duplicates

def compare_frame_files(file1, file2, json1, json2):
    """
    Compare two HDF5 noise pattern files by printing frame numbers and plotting timestamps
    for frames that were marked as 'new' (is_new_frame == 1).
    Also compute average and standard deviation of timestamp differences.
    """

    # load json files
     # --- Load experiment metadata ---
    with open(json1, "r", encoding="utf-8") as f1, open(json2, "r", encoding="utf-8") as f2:
        config1 = json.load(f1)
        config2 = json.load(f2)

    trigger_ns_1 = config1["arduino"]["hardware_trigger_2p_time"]
    trigger_ns_2 = config2["arduino"]["hardware_trigger_2p_time"]   

    # Load both files
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        fn1 = f1['frame_numbers'][:]
        ts1 = f1['timestamps'][:]
        new1 = f1['is_new_frame'][:]

        fn2 = f2['frame_numbers'][:]
        ts2 = f2['timestamps'][:]
        new2 = f2['is_new_frame'][:]

    # Restrict to NEW frames only
    fn1_new = fn1[new1 == 1]
    ts1_new = ts1[new1 == 1]
    fn2_new = fn2[new2 == 1]
    ts2_new = ts2[new2 == 1]

    # Make sure we only compare up to the shorter set of NEW frames
    n = min(len(fn1_new), len(fn2_new))
    fn1_new, ts1_new = fn1_new[:n], ts1_new[:n]
    fn2_new, ts2_new = fn2_new[:n], ts2_new[:n]

    # Compute relative timestamps (ms) with respect to the trigger time
    t0_1, t0_2 = trigger_ns_1, trigger_ns_2
    # t0_1, t0_2 = ts1_new[0], ts2_new[0]
    rel_ts1 = (ts1_new - t0_1) / 1e6  # ms
    rel_ts2 = (ts2_new - t0_2) / 1e6  # ms

    # Compute difference between files for each new frame
    diffs = rel_ts1 - rel_ts2
    avg_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    # Print side-by-side comparison
    print(f"--- New Frame Comparison ({file1} vs {file2}) ---")
    for i in range(n):
        print(f"Idx {i:4d} | File1 Frame {fn1_new[i]:6d}, Time {rel_ts1[i]:10.2f} ms "
              f"| File2 Frame {fn2_new[i]:6d}, Time {rel_ts2[i]:10.2f} ms "
              f"| Δ = {diffs[i]:8.2f} ms")
    print("--- End of new frame comparison ---\n")

    # Print statistics
    print(f"Timestamp difference stats (File1 - File2):")
    print(f"  Average difference: {avg_diff:.2f} ms")
    print(f"  Std deviation:      {std_diff:.2f} ms\n")

     # --- Plot 2: Histogram of timing differences ---
    plt.figure(figsize=(10, 5))
    bins = np.linspace(avg_diff - 3*std_diff, avg_diff + 3*std_diff, 40)
    plt.hist(diffs, bins=bins, color="gray", edgecolor="black", alpha=0.7)
    plt.axvline(avg_diff, color="red", linestyle="--", label=f"Mean = {avg_diff:.4f}ms")
    plt.axvline(avg_diff - std_diff, color="blue", linestyle=":", label=f"±1σ = {std_diff:.4f}ms")
    plt.axvline(avg_diff + std_diff, color="blue", linestyle=":")
    plt.xlabel("Difference in timestamps for the same new frames between experiments (ms)")
    plt.ylabel("Count")
    plt.title("Histogram of Timing Differences for New Frames")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot relative timestamps for new frames
    plt.figure(figsize=(10,5))
    plt.plot(range(n), rel_ts1, 'o-', label=f"{file1}")
    plt.plot(range(n), rel_ts2, 's-', label=f"{file2}")
    plt.xlabel("New frame index")
    plt.ylabel("Relative timestamp (ms)")
    plt.title("New frame timing consistency comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot differences
    plt.figure(figsize=(10,5))
    plt.plot(range(n), diffs, 'x-', color='red', label="Δ (File1 - File2)")
    plt.xlabel("New frame index")
    plt.ylabel("Timestamp difference (ms)")
    plt.title("Frame-by-frame timestamp differences")
    plt.axhline(avg_diff, color='green', linestyle='--', label=f"Mean = {avg_diff:.2f} ms")
    plt.axhline(avg_diff + std_diff, color='orange', linestyle=':', label=f"±1σ = {std_diff:.2f} ms")
    plt.axhline(avg_diff - std_diff, color='orange', linestyle=':')
    plt.legend()
    plt.grid(True)
    plt.show()


def play_new_frame_comparison(file1, file2, fps=2):
    """
    Play side-by-side comparison of new frames from two HDF5 noise pattern files in an OpenCV window.

    Args:
        file1 (str): Path to the first HDF5 file
        file2 (str): Path to the second HDF5 file
        fps (float): Playback frame rate (default: 2 fps for slow viewing)
    """
    delay = int(1000 / fps)  # ms per frame

    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        fn1 = f1['frame_numbers'][:]
        ts1 = f1['timestamps'][:]
        new1 = f1['is_new_frame'][:]

        fn2 = f2['frame_numbers'][:]
        ts2 = f2['timestamps'][:]
        new2 = f2['is_new_frame'][:]

        # Filter only new frames
        fn1_new = fn1[new1 == 1]
        ts1_new = ts1[new1 == 1]
        fn2_new = fn2[new2 == 1]
        ts2_new = ts2[new2 == 1]

        n = min(len(fn1_new), len(fn2_new))
        t0_1, t0_2 = ts1_new[0], ts2_new[0]
        rel_ts1 = (ts1_new - t0_1) / 1e6  # ms
        rel_ts2 = (ts2_new - t0_2) / 1e6  # ms

        cv2.namedWindow("Frame Comparison", cv2.WINDOW_NORMAL)

        paused = False
        idx = 0
        while idx < n:
            fnum1, fnum2 = int(fn1_new[idx]), int(fn2_new[idx])
            img1 = f1['bowl_images'][f'frame_{fnum1:06d}'][:]
            img2 = f2['bowl_images'][f'frame_{fnum2:06d}'][:]

            # Concatenate images horizontally
            combined = np.hstack((img1, img2))

            # Overlay text with frame numbers + relative times
            info_text1 = f"Idx {idx} | F1:{fnum1} @ {rel_ts1[idx]:.1f} ms"
            info_text2 = f"Idx {idx} | F2:{fnum2} @ {rel_ts2[idx]:.1f} ms"

            cv2.putText(combined, info_text1, (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(combined, info_text2, (combined.shape[1]//2 + 10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

            # Show side-by-side image
            cv2.imshow("Frame Comparison", combined)

            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key == 27:  # ESC → quit
                break
            elif key == 32:  # SPACE → toggle pause/resume
                paused = not paused
                continue

            if not paused:
                idx += 1

        cv2.destroyAllWindows()


def subtract_images(file1, file2):
    """
    Compare new frames from two HDF5 noise pattern files by subtracting bowl images.
    Indicate when the images are not identical.
    """
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        fn1 = f1['frame_numbers'][:]
        ts1 = f1['timestamps'][:]
        new1 = f1['is_new_frame'][:]

        fn2 = f2['frame_numbers'][:]
        ts2 = f2['timestamps'][:]
        new2 = f2['is_new_frame'][:]

        # Filter only new frames
        fn1_new = fn1[new1 == 1]
        fn2_new = fn2[new2 == 1]

        n = min(len(fn1_new), len(fn2_new))
        print(f"--- Frame Subtraction Comparison ({file1} vs {file2}) ---")

        for idx in range(n):
            fnum1, fnum2 = int(fn1_new[idx]), int(fn2_new[idx])
            key1 = f'frame_{fnum1:06d}'
            key2 = f'frame_{fnum2:06d}'

            img1 = f1['bowl_images'][key1][:]
            img2 = f2['bowl_images'][key2][:]

            diff = cv2.absdiff(img1, img2)
            nonzero = np.count_nonzero(diff)

            if nonzero > 0:
                print(f"File1:{fnum1:06d} and File2:{fnum2:06d}  -> Difference detected: {nonzero} pixels")

        print("--- End of comparison ---")



def read_handshake_timestamps(filepath):
    """
    Reads an HDF5 file created by noise_pattern_writer and extracts timestamps
    where an Arduino handshake was received.

    Args:
        filepath (str): Path to the .h5 file.

    Returns:
        np.ndarray: Array of timestamps (int64) in nanoseconds.
    """
    with h5py.File(filepath, 'r') as f:
        timestamps = f['timestamps'][:]
        handshake_flags = f['arduino_handshake'][:]

    # Filter where handshake flag == 1
    handshake_timestamps = (timestamps[handshake_flags == 1] - timestamps[0]) / 1e9  # relative to first timestamp in ms

    return handshake_timestamps

def get_first_frame_relative_time(json_path, h5_path):
    """
    Returns the time of the first frame relative to the trigger (in seconds).
    
    Args:
        json_path (str): Path to the JSON config file.
        h5_path (str): Path to the HDF5 file generated by noise_pattern_writer.
    
    Returns:
        float: Time of the first frame relative to trigger in seconds.
    """
    # Load JSON and get trigger timestamp
    with open(json_path, "r") as f:
        config = json.load(f)
    trigger_time = config["arduino"]["hardware_trigger_2p_time"]
    dark_screen_duration = config["dark_screen_duration_s"]

    # Load HDF5 and get first frame timestamp
    with h5py.File(h5_path, "r") as f:
        first_frame_time = f["timestamps"][0]
        second_frame_time = f["timestamps"][1]
        third_frame_time = f["timestamps"][2]

    # Convert nanoseconds difference to seconds
    relative_time_sec = (first_frame_time - trigger_time) / 1e9

    print(f"First frame time: {first_frame_time} ns")
    print(f"Trigger time:     {trigger_time} ns")
    print(f"Time between first and second frame: {(second_frame_time - first_frame_time) /1e9 } seconds")
    print(f"Time between second and third frame: {(third_frame_time - second_frame_time) /1e9 } seconds")
    # Adjust for dark screen duration
    relative_time_sec_dark = relative_time_sec - dark_screen_duration
    
    return relative_time_sec, relative_time_sec_dark
