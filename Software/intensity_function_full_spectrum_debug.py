
import sys
import numpy as np
import rawpy
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def resize_to_fit_screen(img, max_width=1920, max_height=1080):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized_img, scale

def select_line(display_img):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("Select 2 points", display_img)

    cv2.imshow("Select 2 points", display_img)
    cv2.setMouseCallback("Select 2 points", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        raise ValueError("You must select exactly two points.")
    return points

def get_profile_along_line(img, p1, p2):
    x0, y0 = p1
    x1, y1 = p2
    length = int(np.hypot(x1 - x0, y1 - y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    x = np.clip(x.astype(np.int32), 0, img.shape[1] - 1)
    y = np.clip(y.astype(np.int32), 0, img.shape[0] - 1)
    return img[y, x], x, y

def annotate_pixel_positions(ax, x_vals, y_vals, step=50):
    for i in range(0, len(x_vals), step):
        x, y = x_vals[i], y_vals[i]
        ax.text(x, y, str(i), color='white', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

def main(debug=False):
    if len(sys.argv) < 2:
        print("Usage: python script.py path_to_dng [--debug]")
        sys.exit(1)

    dng_path = sys.argv[1]
    with rawpy.imread(dng_path) as raw:
        rgb_image = raw.postprocess(use_camera_wb=True, output_bps=16)

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_norm = (gray - gray.min()) / (gray.max() - gray.min())

    if debug:
        print(f"[DEBUG] Grayscale min: {gray.min()}, max: {gray.max()}, mean: {gray.mean()}")
        
        # Histogram of grayscale image
        plt.figure()
        plt.hist(gray.ravel(), bins=500, color='gray')
        plt.title("Histogram of Grayscale Intensities")
        plt.xlabel("Intensity")
        plt.ylabel("Pixel Count")
        plt.grid(True)
        plt.show()

        # Visualize individual RGB channels
        for i, ch in enumerate(['Red', 'Green', 'Blue']):
            plt.figure()
            plt.imshow(rgb_image[:, :, i], cmap='gray')
            plt.title(f"{ch} Channel (16-bit)")
            plt.colorbar()
            plt.show()

    rgb_display = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb_display_resized, scale = resize_to_fit_screen(rgb_display)
    display_points = select_line(rgb_display_resized.copy())
    original_points = [(int(x / scale), int(y / scale)) for (x, y) in display_points]

    profile, x_vals, y_vals = get_profile_along_line(gray, *original_points)

    if debug:
        print(f"[DEBUG] Profile length: {len(profile)}")
        print(f"[DEBUG] Profile range: min={profile.min()}, max={profile.max()}")

        # Flat region debug
        flat_region = profile[:200]
        print(f"[DEBUG] Flat region (0–200): min={flat_region.min()}, max={flat_region.max()}")

        plt.figure()
        plt.plot(flat_region, color='black')
        plt.title("Intensity Profile in Region [0–200]")
        plt.xlabel("Pixel Position")
        plt.ylabel("Intensity")
        plt.grid(True)
        plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    ax1.plot(profile, color='black')
    ax1.set_title("Grayscale Intensity Profile")
    ax1.set_xlabel("Pixel Position")
    ax1.set_ylabel("Intensity")

    min_idx = np.argmin(profile)
    max_idx = np.argmax(profile)
    min_val = int(profile[min_idx])
    max_val = int(profile[max_idx])

    # Uncomment if you want to annotate min and max
    # ax1.plot(min_idx, min_val, 'go')
    # ax1.annotate(f"Min\n({min_idx}, {min_val})", xy=(min_idx, min_val),
    #              xytext=(min_idx, min_val + 1000),
    #              arrowprops=dict(arrowstyle='->', color='green'),
    #              fontsize=8, color='green', ha='center')

    # ax1.plot(max_idx, max_val, 'ro')
    # ax1.annotate(f"Max\n({max_idx}, {max_val})", xy=(max_idx, max_val),
    #              xytext=(max_idx, max_val + 1000),
    #              arrowprops=dict(arrowstyle='->', color='red'),
    #              fontsize=8, color='red', ha='center')

    ax2.imshow(gray_norm, cmap='viridis')
    ax2.plot(x_vals, y_vals, color='red')
    annotate_pixel_positions(ax2, x_vals, y_vals, step=max(1, len(x_vals)//5))
    ax2.set_title("Heatmap with Pixel Position Overlay")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_mode = "--debug" in sys.argv
    main(debug=debug_mode)
