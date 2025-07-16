import numpy as np
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# Parameters
# -----------------------------
res_x, res_y = 320, 180              # Projector resolution
throw_ratio = 1.4                     # Projector throw ratio
d_proj = 200                          # Distance from fly to projector (in mm or consistent unit)
r = throw_ratio
azimuth_range = (0, 180)              # degrees
elevation_range = (0, 140)            # degrees

# -----------------------------
# Checkerboard Configuration
# -----------------------------
checkerboard_shape = (12, 8)
checkerboard_square_size = 1         # degrees per square (controls angular resolution)

# -----------------------------
# Step 1: Uniform Sampling in Angular Space
# -----------------------------
n_alpha = res_x
n_beta = res_y
alpha = np.deg2rad(np.linspace(azimuth_range[0], azimuth_range[1], n_alpha))
beta = np.deg2rad(np.linspace(elevation_range[0], elevation_range[1], n_beta))
alpha_grid, beta_grid = np.meshgrid(alpha, beta)

# -----------------------------
# Step 2: Surface Function S(α, β)
# -----------------------------
tan_alpha = np.tan(alpha_grid)
denominator = alpha_grid - np.pi * r * tan_alpha

# Avoid divide-by-zero/singularity
denominator = alpha_grid - np.pi * r * tan_alpha

# Avoid divide-by-zero/singularity
epsilon = 1e-3
invalid_mask = np.abs(denominator) < epsilon
denominator_safe = np.where(invalid_mask, np.nan, denominator)

with np.errstate(divide='ignore', invalid='ignore'):
    scalar = (alpha_grid * (1 + 2 * tan_alpha)) / denominator_safe
    scalar *= d_proj
    scalar[invalid_mask] = np.nan


X = scalar
Y = scalar * np.cos(beta_grid)
Z = scalar * np.sin(beta_grid)
points_3D = np.stack([X, Y, Z], axis=-1)  # shape: (H, W, 3)

# -----------------------------
# Step 3: Project to Projector Plane (Pinhole model)
# -----------------------------
# Translate to projector frame (projector is at y = d_proj)
points_cam = points_3D.copy()
points_cam[:, :, 1] -= d_proj

# Perspective projection
eps = 1e-6
px = points_cam[:, :, 0] * d_proj / (points_cam[:, :, 1] + eps)
py = points_cam[:, :, 2] * d_proj / (points_cam[:, :, 1] + eps)

# Normalize to image coordinates
u = (px - np.nanmin(px)) / (np.nanmax(px) - np.nanmin(px)) * (res_x - 1)
v = (py - np.nanmin(py)) / (np.nanmax(py) - np.nanmin(py)) * (res_y - 1)

# Mask out NaNs
u = np.nan_to_num(u, nan=0).astype(np.float32)
v = np.nan_to_num(v, nan=0).astype(np.float32)

# -----------------------------
# Step 4: Create Angular Checkerboard
# -----------------------------
# Create angular checkerboard pattern
check = (
    ((np.floor(alpha_grid / np.deg2rad(checkerboard_square_size)).astype(int) +
      np.floor(beta_grid / np.deg2rad(checkerboard_square_size)).astype(int)) % 2) * 255
).astype(np.uint8)

# -----------------------------
# Step 5: Warp Checkerboard to Projector Space
# -----------------------------
map_x = u
map_y = v
warped_checkerboard = cv2.remap(check, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# -----------------------------
# Step 6: Display on Second Monitor (Projector)
# -----------------------------
window_name = "Warped Checkerboard"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, 0, 0)  # Move to projector screen (assumed right of primary)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(window_name, warped_checkerboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
