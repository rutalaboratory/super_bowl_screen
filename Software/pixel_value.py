import rawpy
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# Load DNG image
filename = sys.argv[1]
with rawpy.imread(filename) as raw:
    raw_image = raw.raw_image.copy()
    rgb_image = raw.postprocess()

# Convert to OpenCV format (BGR)
display_image_base = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# Get screen resolution (fallback to 1080p if unavailable)
screen_width = 1920
screen_height = 1080
try:
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
except:
    pass

# Resize image to fit screen while preserving aspect ratio
img_h, img_w = display_image_base.shape[:2]
scale = min(screen_width / img_w, screen_height / img_h, 1.0)
resized_display = cv2.resize(display_image_base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# Let user select ROI on scaled image
roi = cv2.selectROI("Select ROI", resized_display, fromCenter=False, showCrosshair=True)
x, y, w, h = [int(v / scale) for v in roi]  # Scale ROI back to raw coordinates
cv2.destroyWindow("Select ROI")

# Extract raw ROI and print values
raw_roi = raw_image[y:y+h, x:x+w]
print("All raw values in ROI (row-major order):")
print(raw_roi)

# Prepare for mouse click interaction
display_image = display_image_base.copy()

def mouse_callback(event, x, y, flags, param):
    global display_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if 0 <= y < raw_image.shape[0] and 0 <= x < raw_image.shape[1]:
            raw_value = int(raw_image[y, x])
            print(f"Raw value at ({x}, {y}): {raw_value}")
            display_image = display_image_base.copy()  # Clear previous markings

            # Highlight exactly one pixel by changing its color
            display_image[y, x] = (0, 0, 255)  # Red pixel

            # Draw value nearby
            cv2.putText(display_image, f"{raw_value}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Display full-resolution image for interaction
cv2.namedWindow('DNG Viewer', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('DNG Viewer', mouse_callback)

while True:
    cv2.imshow('DNG Viewer', display_image)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
