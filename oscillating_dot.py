import cv2
import numpy as np
import argparse
import time
from screeninfo import get_monitors

# ----------------------
# Parse command-line arguments
# ----------------------
parser = argparse.ArgumentParser(description="Project a moving dot on second monitor.")
parser.add_argument("--radius", type=int, default=20, help="Dot radius in pixels")
parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
parser.add_argument("--speed", type=float, default=10, help="Speed of the dot in seconds")
args = parser.parse_args()

dot_radius = args.radius
duration = args.duration
speed = args.speed
# ----------------------
# Get second monitor
# ----------------------
monitors = get_monitors()
if len(monitors) < 2:
    raise RuntimeError("Second monitor not found.")
monitor = monitors[1]
screen_width, screen_height = monitor.width, monitor.height
x_offset, y_offset = monitor.x, monitor.y

# ----------------------
# Create fullscreen window on second monitor
# ----------------------
window_name = "Click to select bounds and vertical position"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, x_offset, y_offset)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

click_positions = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(click_positions) < 3:
        click_positions.append((x, y))

cv2.setMouseCallback(window_name, mouse_callback)

# ----------------------
# Prompt user to click 3 points
# ----------------------
print("Click 1: MIN horizontal bound")
print("Click 2: MAX horizontal bound")
print("Click 3: Vertical position of the dot")

while len(click_positions) < 3:
    frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        cv2.destroyAllWindows()
        exit()

# Extract and sort input values
x_min = min(click_positions[0][0], click_positions[1][0])
x_max = max(click_positions[0][0], click_positions[1][0])
dot_y = click_positions[2][1]

print(f"Selected horizontal bounds: min = {x_min}, max = {x_max}")
print(f"Selected vertical position: y = {dot_y}")

# ----------------------
# Start dot animation
# ----------------------
x_pos = x_min
direction = 1
# speed = (x_max - x_min) / 2  # pixels/sec

start_time = time.time()
last_time = start_time

while True:
    now = time.time()
    dt = now - last_time
    elapsed = now - start_time
    last_time = now

    if elapsed >= duration:
        break

    # Update horizontal position
    x_pos += direction * speed * dt
    if x_pos >= x_max:
        x_pos = x_max
        direction = -1
    elif x_pos <= x_min:
        x_pos = x_min
        direction = 1

    # Draw
    frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
    cv2.circle(frame, (int(x_pos), int(dot_y)), dot_radius, (0, 0, 0), -1)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
