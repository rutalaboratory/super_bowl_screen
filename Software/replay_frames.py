import h5py
import cv2
import numpy as np
import time
import sys
import os
from datetime import datetime


def replay_frames(h5_path, playback_speed=1.0):
    """
    Replay frames from an HDF5 file with timestamps and frame numbers overlaid.
    
    Args:
        h5_path: Path to the HDF5 file containing frames
        playback_speed: Speed multiplier (1.0 = real time, 2.0 = 2x speed, etc.)
    """
    print(f"Opening {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        frames_group = f['frames']
        timestamps = f['timestamps'][:]
        frame_numbers = f['frame_numbers'][:]
        
        total_frames = len(frame_numbers)
        print(f"Found {total_frames} frames")
        
        # Create window
        cv2.namedWindow('Replay', cv2.WINDOW_NORMAL)
        
        # Calculate time deltas between frames for playback timing
        time_deltas = np.diff(timestamps)
        time_deltas = np.append(time_deltas, time_deltas[-1])  # Add last delta
        
        start_time = time.time()
        last_frame_time = start_time
        
        for i in range(total_frames):
            # Read frame
            frame = frames_group[f'frame_{i:06d}'][:]
            timestamp = timestamps[i]
            frame_number = frame_numbers[i]
            
            # Add text overlay
            text = f"Frame: {frame_number}  Time: {timestamp:.3f}s"
            frame_with_text = frame.copy()
            
            # Add black background for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame_with_text, (10, 20), (10 + text_size[0], 20 + text_size[1] + 10), 
                         (0, 0, 0), -1)
            
            # Add text
            cv2.putText(frame_with_text, text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Replay', frame_with_text)
            
            # Calculate wait time for proper playback speed
            target_time = start_time + (timestamp - timestamps[0]) / playback_speed
            current_time = time.time()
            wait_time = max(1, int((target_time - current_time) * 1000))
            
            # Wait and check for quit
            key = cv2.waitKey(wait_time)
            if key == ord('q') or key == 27:  # q or ESC to quit
                break
            
            # Print FPS every 60 frames
            if i % 60 == 0:
                actual_fps = 60 / (time.time() - last_frame_time)
                print(f"Playback FPS: {actual_fps:.2f}")
                last_frame_time = time.time()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay_frames.py <path_to_h5_file> [playback_speed]")
        print("Example: python replay_frames.py noise_patterns/noise_patterns_20250821.h5 1.0")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    playback_speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    if not os.path.exists(h5_path):
        print(f"Error: File {h5_path} not found")
        sys.exit(1)
        
    replay_frames(h5_path, playback_speed)
