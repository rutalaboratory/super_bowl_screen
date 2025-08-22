import rawpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
import os
import sys

def process_dng_to_heatmap_and_surface(filepath, zoom_factor=3):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read DNG using rawpy
    with rawpy.imread(filepath) as raw:
        raw_image = raw.raw_image_visible
        height, width = raw_image.shape

        # Normalize the raw data between 0 and 1
        raw_min = np.min(raw_image)
        raw_max = np.max(raw_image)
        print("Min RAW:", raw_min)
        print("Max RAW:", raw_max)

        normalized = (raw_image - raw_min) / (raw_max - raw_min)

        # Metadata
        print(f"Channels: 1 (grayscale raw image)")
        print(f"Size of each channel: {raw_image.size} pixels")

        # Heatmap
        plt.figure()
        plt.imshow(normalized, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Normalized Intensity (0–1)')
        plt.title("Heatmap")
        plt.axis('off')
        plt.show()

        # Save heatmap
        heatmap_output_path = os.path.splitext(filepath)[0] + "_heatmap.png"
        plt.imsave(heatmap_output_path, normalized, cmap='viridis')
        print(f"Saved heatmap to: {heatmap_output_path}")

        # Smooth the data for the surface plot
        smoothed = zoom(normalized, zoom_factor)

        # Create meshgrid for smoothed dimensions
        smoothed_height, smoothed_width = smoothed.shape
        x = np.linspace(0, width, smoothed_width)
        y = np.linspace(0, height, smoothed_height)
        X, Y = np.meshgrid(x, y)
        Z = smoothed

        # 3D Surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Normalized Intensity (0–1)')
        ax.set_title("Smoothed 3D Surface Plot of Intensity")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Intensity")
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dng_heatmap_surface.py <path_to_dng_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    process_dng_to_heatmap_and_surface(filepath)
