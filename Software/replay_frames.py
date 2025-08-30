#!/usr/bin/env python3
"""
replay_noise_patterns.py

Replay an HDF5 recording produced by noise_pattern_writer:
- Displays the 'original_images' image and the 'bowl_images' image side-by-side
- Overlays frame number and human-readable timestamp on each panel
- Uses recorded timestamps to approximate original playback timing (unless --fps is specified)

Controls:
  q or ESC  : quit
  space     : pause/resume
  → (right) : step forward one frame (when paused)
  ← (left)  : step back one frame (when paused)
  up/down   : increase/decrease playback speed (realtime mode only)

Requirements: h5py, numpy, opencv-python
"""

import argparse
import time
from datetime import datetime
import h5py
import numpy as np
import cv2

def to_bgr(img):
    """Ensure image is 3-channel BGR for consistent overlay."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[:, :, :3]
    return np.stack([arr] * 3, axis=-1) if arr.ndim == 2 else arr

def put_label(img, text, org=(8, 24), scale=0.6, color=(255, 255, 255), bg=(0, 0, 0)):
    """Draw text with a simple background box for readability."""
    if img is None:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x-4, y-th-4), (x+tw+4, y+baseline+4), bg, thickness=-1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def format_ts(ts_float, tz_local=True):
    """Format a POSIX timestamp float into readable local time."""
    try:
        dt = datetime.fromtimestamp(float(ts_float)) if tz_local else datetime.utcfromtimestamp(float(ts_float))
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except Exception:
        return f"{ts_float:.6f}"

def resize_to_match(img, target_h):
    """Resize image to target height keeping aspect ratio."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(round(w * (target_h / float(h))))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def build_frame(fr_img, out_img, fn, ts, scale=1.0):
    """Compose a side-by-side frame with overlays."""
    fr = to_bgr(fr_img)
    out = to_bgr(out_img)
    if fr is None or out is None:
        return None

    # Match heights for clean hstack
    if fr.shape[0] != out.shape[0]:
        out = resize_to_match(out, fr.shape[0])

    # Overlay labels
    put_label(fr, f"Frame: {fn}")
    put_label(fr, f"Time: {format_ts(ts)}", org=(8, 48))
    put_label(out, f"Frame: {fn}")
    put_label(out, f"Time: {format_ts(ts)}", org=(8, 48))

    combo = np.hstack([fr, out])

    if scale != 1.0:
        combo = cv2.resize(
            combo,
            (int(combo.shape[1] * scale), int(combo.shape[0] * scale)),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
    return combo

def read_index_datasets(h5f):
    """Read timestamps and frame_numbers arrays (required)."""
    if "timestamps" not in h5f or "frame_numbers" not in h5f:
        raise KeyError("HDF5 must contain 'timestamps' and 'frame_numbers' datasets.")
    ts = np.array(h5f["timestamps"])
    fnums = np.array(h5f["frame_numbers"])
    return ts, fnums

def get_frame_datasets(h5f, frame_number):
    """
    Fetch the per-frame datasets saved by noise_pattern_writer.
    """
    ds_name = f"frame_{int(frame_number):06d}"
    orig_grp = h5f["original_images"]
    bowl_grp = h5f["bowl_images"]

    if ds_name not in orig_grp or ds_name not in bowl_grp:
        return None, None

    fr = orig_grp[ds_name][()]
    out = bowl_grp[ds_name][()]
    return fr, out

def clamp(x, lo, hi):
    """Clamp value between lo and hi."""
    return max(lo, min(hi, x))

def main():
    ap = argparse.ArgumentParser(description="Replay HDF5 noise pattern recording.")
    ap.add_argument("h5", help="Path to noise_patterns_*.h5")
    ap.add_argument("--window", default="Noise Replay", help="Window title")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor for display (e.g., 0.75)")
    ap.add_argument("--fps", type=float, default=0.0, help="Override FPS for timing.")
    ap.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (only when --fps=0).")
    ap.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    ap.add_argument("--end", type=int, default=-1, help="End index (exclusive). -1 = till end")
    ap.add_argument("--loop", action="store_true", help="Loop playback")
    args = ap.parse_args()

    print(f"[Replay] Opening {args.h5}")
    with h5py.File(args.h5, "r") as f:
        if "original_images" not in f or "bowl_images" not in f:
            raise KeyError("HDF5 must contain 'original_images' and 'bowl_images' groups.")

        timestamps, frame_numbers = read_index_datasets(f)
        n = len(timestamps)
        if len(frame_numbers) != n:
            raise ValueError("Length mismatch between 'timestamps' and 'frame_numbers'.")

        # Determine range
        start = clamp(args.start, 0, n-1)
        end = n if args.end < 0 else clamp(args.end, 0, n)
        if start >= end:
            print("Nothing to play (empty range).")
            return

        # OpenCV window (guard against unavailable GUI flags)
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
            flags |= cv2.WINDOW_GUI_EXPANDED
        elif hasattr(cv2, "WINDOW_GUI_NORMAL"):
            flags |= cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(args.window, flags)

        idx = start
        paused = False
        speed = max(0.05, args.speed)  # avoid zero/negative
        fixed_delay_ms = int(1000.0 / args.fps) if args.fps > 0 else None

        while True:
            # Build frame image
            ts = timestamps[idx]
            fn = int(frame_numbers[idx])

            fr, out = get_frame_datasets(f, fn)
            if fr is None or out is None:
                # Missing dataset(s). Skip ahead.
                print(f"[Replay] Missing datasets for frame_number={fn}; skipping.")
                idx += 1
                if idx >= end:
                    if args.loop:
                        idx = start
                        continue
                    break
                continue

            frame_img = build_frame(fr, out, fn, ts, scale=args.scale)
            if frame_img is None:
                print(f"[Replay] Failed to compose frame idx={idx} (fn={fn}); skipping.")
                idx += 1
                if idx >= end:
                    if args.loop:
                        idx = start
                        continue
                    break
                continue

            cv2.imshow(args.window, frame_img)

            # Determine delay (robust to NaN/Inf/negative dt)
            if paused:
                delay_ms = 50  # responsive while paused
            else:
                if fixed_delay_ms is not None:
                    delay_ms = max(1, fixed_delay_ms)
                else:
                    # Realtime based on timestamp delta
                    if idx + 1 < end:
                        t0 = float(timestamps[idx])
                        t1 = float(timestamps[idx + 1])
                        # sanitize: handle NaN/Inf and negative or absurd deltas
                        if not np.isfinite(t0) or not np.isfinite(t1):
                            dt = 0.04
                        else:
                            dt = t1 - t0
                            if not np.isfinite(dt) or dt < 0:
                                dt = 0.04
                            # clamp to a reasonable range to avoid zero/huge waits
                            dt = float(np.clip(dt, 0.02, 5.0))
                        delay_ms = max(1, int(1000 * dt * speed))
                    else:
                        delay_ms = 50  # last frame

            # Handle key press
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q
                break
            elif key == ord(' '):  # Pause toggle
                paused = not paused
            elif key == 81 or key == 2424832:  # Left arrow (step back)
                idx = max(start, idx - 1)
            elif key == 83 or key == 2555904:  # Right arrow (step forward)
                idx = min(end - 1, idx + 1)
            elif key == 82 or key == 2490368:  # Up arrow (increase speed)
                speed *= 1.1
            elif key == 84 or key == 2621440:  # Down arrow (decrease speed)
                speed /= 1.1

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
