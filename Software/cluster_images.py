import rawpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import leastsq
import sys

def fit_circle_scipy(points):
    points = np.array(points)
    x_m, y_m = np.mean(points, axis=0)

    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f_2b(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_opt, _ = leastsq(f_2b, center_estimate)
    Ri = calc_R(*center_opt)
    R_opt = Ri.mean()

    center = tuple(map(int, center_opt))
    radius = int(R_opt)
    return center, radius

def create_circle_mask(shape, center, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask

# --- Step 1: Load raw image ---
dng_path = sys.argv[1]

with rawpy.imread(dng_path) as raw:
    raw_image = raw.raw_image_visible.astype(float)
    # --- Min/max for entire raw image ---
    raw_min = np.min(raw_image)
    raw_max = np.max(raw_image)
    print(f"Full raw image: min = {raw_min}, max = {raw_max}")

    rgb = raw.postprocess()

# --- Step 2: First circle ---
plt.imshow(rgb)
plt.title("Select points for FIRST circle. Press Enter when done.")
pts1 = plt.ginput(n=-1, timeout=0)
plt.close()

center1, radius1 = fit_circle_scipy(pts1)
print(f"First circle: center={center1}, radius={radius1}")

# --- Step 3: Second circle ---
plt.imshow(rgb)
plt.title("Select points for SECOND circle. Press Enter when done.")
pts2 = plt.ginput(n=-1, timeout=0)
plt.close()

center2, radius2 = fit_circle_scipy(pts2)
print(f"Second circle: center={center2}, radius={radius2}")

# --- Step 4: Create masks ---
mask1 = create_circle_mask(rgb.shape, center1, radius1)
mask2 = create_circle_mask(rgb.shape, center2, radius2)
final_mask = np.logical_and(mask1 == 1, mask2 == 0)

# --- Step 5: Histogram of raw image (before and after masking) ---
raw_all = raw_image.flatten()
raw_region = raw_image[final_mask]

# --- Step 5: Histogram of raw image (before and after masking) ---
raw_all = raw_image.flatten()
raw_region = raw_image[final_mask]

# Compute histograms first to get common y-limit
counts_all, bins_all = np.histogram(raw_all, bins=200)
counts_region, bins_region = np.histogram(raw_region, bins=200)
max_count = max(np.max(counts_all), np.max(counts_region))

plt.figure(figsize=(14, 5))

# Left: All pixels
plt.subplot(1, 2, 1)
plt.hist(raw_all, bins=200, color='gray', alpha=0.8)
plt.yscale('log')
plt.ylim(1, max_count * 1.1)
plt.xlabel("Raw pixel value")
plt.ylabel("Pixel count (log scale)")
plt.title("Histogram: All Raw Image Pixels")

# Right: Selected region
plt.subplot(1, 2, 2)
plt.hist(raw_region, bins=200, color='blue', alpha=0.8)
plt.yscale('log')
plt.ylim(1, max_count * 1.1)
plt.xlabel("Raw pixel value")
plt.ylabel("Pixel count (log scale)")
plt.title("Histogram: Selected Region Only")

plt.tight_layout()
plt.show()

# --- Step 6: Min/max in region ---
min_val = np.min(raw_region)
max_val = np.max(raw_region)
print(f"\nMin raw value in final region: {min_val}")
print(f"Max raw value in final region: {max_val}")

# --- Step 7: Display masked RGB image ---
rgb_masked = rgb.astype(float)
for c in range(3):
    channel = rgb_masked[..., c]
    channel[~final_mask] = np.nan
    rgb_masked[..., c] = channel

plt.imshow(rgb_masked / 255)
plt.title("Final Region (First circle minus intersection with second)")
plt.axis('off')
plt.show()
