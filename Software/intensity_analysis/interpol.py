
import rawpy
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import RectangleSelector
from scipy.interpolate import griddata

class DNGHeatmapExplorer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.normalized = None
        self.fig, self.ax = plt.subplots()
        self.rect_selector = None
        self.selected_coords = None

    def load_and_normalize(self):
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        with rawpy.imread(self.filepath) as raw:
            raw_image = raw.raw_image_visible
            raw_min, raw_max = np.min(raw_image), np.max(raw_image)
            print(f"Min RAW: {raw_min}, Max RAW: {raw_max}")
            self.normalized = (raw_image - raw_min) / (raw_max - raw_min)
            print(f"Image shape: {self.normalized.shape}")

    def display_image(self):
        self.ax.imshow(self.normalized, cmap='viridis', interpolation='nearest')
        self.ax.set_title("Heatmap - Draw rectangle to smooth")
        self.ax.axis('off')
        self.fig.colorbar(self.ax.images[0], ax=self.ax, label='Normalized Intensity (0–1)')

        self.rect_selector = RectangleSelector(
    self.ax, self.on_select, useblit=True,
    button=[1], minspanx=5, minspany=5, interactive=True)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def on_select(self, selector):
        x_min, x_max, y_min, y_max = map(int, selector.extents)
        print(f"Selected area: ({x_min}, {y_min}) to ({x_max}, {y_max})")

        roi = self.normalized[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            print("Empty selection.")
            return

        # Smooth using interpolation
        grid_y, grid_x = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        points = np.array([[y, x] for y in range(roi.shape[0]) for x in range(roi.shape[1])])
        values = roi.flatten()
        smooth_roi = griddata(points, values, (grid_y, grid_x), method='linear')

        self.normalized[y_min:y_max, x_min:x_max] = smooth_roi
        self.ax.images[0].set_data(self.normalized)
        self.fig.canvas.draw()


        # Smooth using interpolation
        grid_y, grid_x = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        points = np.array([[y, x] for y in range(roi.shape[0]) for x in range(roi.shape[1])])
        values = roi.flatten()
        smooth_roi = griddata(points, values, (grid_y, grid_x), method='linear')

        self.normalized[y1:y2, x1:x2] = smooth_roi
        self.ax.images[0].set_data(self.normalized)
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < self.normalized.shape[1] and 0 <= y < self.normalized.shape[0]:
            intensity = self.normalized[y, x]
            print(f"Clicked at ({x}, {y}) - Intensity: {intensity:.4f}")

if __name__ == "__main__":
    filepath = input("Enter the path to a DNG file: ").strip()
    explorer = DNGHeatmapExplorer(filepath)
    explorer.load_and_normalize()
    explorer.display_image()
