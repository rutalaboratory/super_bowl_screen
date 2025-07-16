import rawpy
import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import sys

def load_dng_image(filepath):
    """Load a DNG image and return RGB."""
    with rawpy.imread(filepath) as raw:
        rgb = raw.postprocess()
    return rgb

def polar_to_cartesian(center, radius, angle_degrees):
    """Convert polar to cartesian coordinates."""
    angle_radians = np.deg2rad(angle_degrees)
    x = center[0] + radius * np.cos(angle_radians)
    y = center[1] + radius * np.sin(angle_radians)
    return int(round(x)), int(round(y))

def draw_annular_sector(ax, center, r_inner, r_outer, theta_start, theta_end, resolution=300):
    """Draw annular sector on image."""
    theta = np.linspace(np.deg2rad(theta_start), np.deg2rad(theta_end), resolution)
    outer_arc = np.array([polar_to_cartesian(center, r_outer, np.rad2deg(t)) for t in theta])
    inner_arc = np.array([polar_to_cartesian(center, r_inner, np.rad2deg(t)) for t in theta[::-1]])
    arc_points = np.vstack((outer_arc, inner_arc, outer_arc[0]))
    ax.plot(arc_points[:, 0], arc_points[:, 1], color='red', linewidth=2)
    ax.fill(arc_points[:, 0], arc_points[:, 1], color='red', alpha=0.2)

def calculate_angle(center, p1, p2):
    """Calculate angle between two vectors from center to p1 and p2."""
    v1 = np.array(p1) - np.array(center)
    v2 = np.array(p2) - np.array(center)
    dot = np.dot(v1, v2)
    det = np.cross(v1, v2)
    angle = np.arctan2(det, dot)
    return np.degrees(angle) % 360

def unwrap_annular_sector(image, center, r_inner, r_outer, theta_start, theta_end, radial_res=300, angular_res=300):
    """Unwrap annular sector into normalized Mono image (float32 [0, 1])."""
    radius_range = np.linspace(r_inner, r_outer, radial_res)
    angle_range = np.linspace(theta_start, theta_end, angular_res)

    unwrapped = np.zeros((radial_res, angular_res), dtype=np.float32)

    for i, r in enumerate(radius_range):
        for j, theta in enumerate(angle_range):
            x, y = polar_to_cartesian(center, r, theta)
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                pixel = image[int(y), int(x)]
                gray = float(np.dot(pixel, [0.2989, 0.5870, 0.1140]))  # Grayscale
                unwrapped[i, j] = gray

    # Normalize to range 0-1
    min_val, max_val = np.min(unwrapped), np.max(unwrapped)
    if max_val > min_val:
        unwrapped = (unwrapped - min_val) / (max_val - min_val)

    return unwrapped


def main():
    parser = argparse.ArgumentParser(description="Unwrap annular sector from a 360-degree camera image.")
    parser.add_argument("filepath", help="Path to the .dng image")
    args = parser.parse_args()

    try:
        image = load_dng_image(args.filepath)
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(1)

    # Step 1: User selects points
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Click 5 Points: Center, Inner Radius, Outer Radius, Start Line, End Line")
    points = plt.ginput(5, timeout=0)
    plt.close(fig)

    if len(points) != 5:
        print("Error: You must select exactly 5 points.")
        return

    center = points[0]
    r_inner = np.linalg.norm(np.array(points[1]) - np.array(center))
    r_outer = np.linalg.norm(np.array(points[2]) - np.array(center))
    
    # Compute angles from center to start and end points
    angle_start = np.degrees(np.arctan2(points[3][1] - center[1], points[3][0] - center[0]))
    angle_end = np.degrees(np.arctan2(points[4][1] - center[1], points[4][0] - center[0]))

    # Normalize to [0, 360)
    angle_start = angle_start % 360
    angle_end = angle_end % 360

    # Compute angular span and center angle based on selection order
    angle_start = angle_start % 360
    angle_end = angle_end % 360

    # Handle wraparound to preserve user direction
    if angle_end < angle_start:
        angle_end += 360  # ensures correct direction

    angle_diff = angle_end - angle_start
    angle_mid = (angle_start + angle_diff / 2) % 360

    theta_start = angle_start
    theta_end = angle_end

    # Step 2: Visualize sector
    fig, ax = plt.subplots()
    ax.imshow(image)
    draw_annular_sector(ax, center, r_inner, r_outer, theta_start, theta_end)
    ax.set_title(f"Sector: Center {angle_mid:.2f}°, Width {angle_diff:.2f}°")
    plt.show()

    # Step 3: Unwrap and display as heatmap
    unwrapped = unwrap_annular_sector(image, center, r_inner, r_outer, theta_start, theta_end)
    plt.figure()
    plt.imshow(unwrapped, cmap='hot', aspect='auto',
           extent=[-angle_diff / 2, angle_diff / 2, r_outer, r_inner])
    plt.colorbar(label='Normalized Intensity (0–1)')
    plt.title("Unwrapped Annular Sector (Mono Heatmap)")
    plt.xlabel("Angle relative to center (degrees)")
    plt.ylabel("Radius (pixels)")
    plt.gca().invert_yaxis()
    plt.show()

    # Step 0: Print image info
    print(f"Image resolution: {image.shape[1]} x {image.shape[0]} (width x height)")
    print(f"Image dtype: {image.dtype}")

    # Get bit range
    bit_depth = image.dtype.itemsize * 8
    print(f"Approximate bit depth per channel: {bit_depth}-bit")

    min_val = image.min()
    max_val = image.max()
    print(f"Pixel value range: [{min_val}, {max_val}]")

if __name__ == "__main__":
    main()
