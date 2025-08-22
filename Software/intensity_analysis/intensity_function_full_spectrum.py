import sys
import numpy as np
import rawpy
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import matplotlib.cm as cmx
import matplotlib.colors as mcolors

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

def get_profile_along_line(img, p1, p2, pixel_width=1):
    x0, y0 = p1
    x1, y1 = p2
    length = int(np.hypot(x1 - x0, y1 - y0))
    x_line, y_line = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    profile = []

    dx = x1 - x0
    dy = y1 - y0
    norm = np.hypot(dx, dy)
    if norm == 0:
        raise ValueError("Points are identical; cannot define a line.")
    dx /= norm
    dy /= norm
    perp_dx = -dy
    perp_dy = dx

    half_width = pixel_width // 2

    for xi, yi in zip(x_line, y_line):
        intensities = []
        for offset in range(-half_width, half_width + 1):
            px = int(round(xi + offset * perp_dx))
            py = int(round(yi + offset * perp_dy))
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                intensities.append(img[py, px])
        if intensities:
            profile.append(np.mean(intensities))
        else:
            profile.append(0.0)

    x_line = np.clip(x_line.astype(np.int32), 0, img.shape[1] - 1)
    y_line = np.clip(y_line.astype(np.int32), 0, img.shape[0] - 1)
    return np.array(profile), x_line, y_line

def annotate_pixel_positions(ax, x_vals, y_vals, step=50):
    for i in range(0, len(x_vals), step):
        x, y = x_vals[i], y_vals[i]
        ax.text(x, y, str(i), color='white', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

def annotate_cm_positions(ax, x_vals, y_vals, cm_vals, step_cm=2.0):
    total_cm = cm_vals[-1]
    for cm in np.arange(0, total_cm, step_cm):
        idx = np.argmin(np.abs(cm_vals - cm))
        x, y = x_vals[idx], y_vals[idx]
        ax.text(x, y, f"{cm:.1f} cm", color='white', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

def get_offset_points(p1, p2, offset, side_index):
    x0, y0 = p1
    x1, y1 = p2
    dx, dy = x1 - x0, y1 - y0
    norm = np.hypot(dx, dy)
    if norm == 0:
        raise ValueError("Identical points")
    dx /= norm
    dy /= norm
    perp_dx = -dy
    perp_dy = dx
    shift_x = perp_dx * offset * side_index
    shift_y = perp_dy * offset * side_index
    return (int(x0 + shift_x), int(y0 + shift_y)), (int(x1 + shift_x), int(y1 + shift_y))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dng_path", help="Path to DNG file")
    parser.add_argument("--shape", choices=["bowl", "cone", "none"], default="none",
                        help="Shape of the surface: 'bowl', 'cone', or 'none'")
    parser.add_argument("--pixel_widths", type=str, default="1",
                        help="Comma-separated pixel widths to average over (e.g., 1,3,5)")
    parser.add_argument("--num_lines_each_side", type=int, default=0,
                        help="Number of parallel lines on each side of the selected line")
    parser.add_argument("--offset_distance", type=float, default=10.0,
                        help="Distance in pixels between parallel lines")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pixel_widths = [int(w.strip()) for w in args.pixel_widths.split(",")]
    if args.shape == "bowl":
        max_length_cm = 17.5
        use_cm = True
    elif args.shape == "cone":
        max_length_cm = 7.0
        use_cm = True
    else:
        max_length_cm = None
        use_cm = False

    with rawpy.imread(args.dng_path) as raw:
        rgb_image = raw.postprocess(use_camera_wb=True, output_bps=16)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_norm = (gray - gray.min()) / (gray.max() - gray.min())
    rgb_display = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb_display_resized, scale = resize_to_fit_screen(rgb_display)
    display_points = select_line(rgb_display_resized.copy())
    original_points = [(int(x / scale), int(y / scale)) for (x, y) in display_points]

    all_profiles = []
    line_segments = []

    total_lines = args.num_lines_each_side * 2 + 1
    jet = plt.get_cmap('jet')
    cNorm = mcolors.Normalize(vmin=0, vmax=total_lines - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for idx, offset_index in enumerate(range(-args.num_lines_each_side, args.num_lines_each_side + 1)):
        p1_offset, p2_offset = get_offset_points(original_points[0], original_points[1],
                                                 args.offset_distance, offset_index)
        line_segments.append((p1_offset, p2_offset))

        for pw in pixel_widths:
            profile, x_vals, y_vals = get_profile_along_line(gray, p1_offset, p2_offset, pixel_width=pw)
            all_profiles.append((pw, offset_index, profile, x_vals, y_vals, scalarMap.to_rgba(idx)))

    if use_cm:
        pixel_length = len(all_profiles[0][2])
        distance_cm = np.linspace(0, max_length_cm, pixel_length)
        if args.debug:
            print(f"Shape: {args.shape}, Max cm: {max_length_cm}, #pixels: {pixel_length}")
            print(f"cm/pixel = {max_length_cm / pixel_length:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    for pw, offset_idx, profile, _, _, color in all_profiles:
        label = f"{pw}px, offset {offset_idx}"
        if use_cm:
            ax1.plot(distance_cm, profile, label=label, color=color)
        else:
            ax1.plot(profile, label=label, color=color)

    ax1.set_title("Grayscale Intensity Profiles")
    ax1.set_xlabel("Distance (cm)" if use_cm else "Pixel Position")
    ax1.set_ylabel("Intensity")
    ax1.legend(fontsize=8)

    ax2.imshow(gray_norm, cmap='viridis')
    for _, _, _, x_vals, y_vals, color in all_profiles:
        ax2.plot(x_vals, y_vals, color=color, linewidth=1.5)

    scale_length_px = 100
    x0, y0 = all_profiles[0][3][0], all_profiles[0][4][0]
    y_bar = min(y0 + 100, gray.shape[0] - 1)
    x_start = int(x0)
    x_end = int(x0 + scale_length_px)
    if x_end < gray.shape[1]:
        ax2.plot([x_start, x_end], [y_bar, y_bar], color='white', linewidth=2)
        ax2.text((x_start + x_end) // 2, y_bar + 40, f"{scale_length_px} px", color='white',
                 ha='center', va='bottom', fontsize=9,
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    if use_cm:
        annotate_cm_positions(ax2, all_profiles[0][3], all_profiles[0][4], distance_cm)
    else:
        annotate_pixel_positions(ax2, all_profiles[0][3], all_profiles[0][4])

    ax2.set_title("Heatmap with Parallel Line Overlays")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
