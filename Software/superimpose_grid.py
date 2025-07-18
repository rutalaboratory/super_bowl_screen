import cv2
import numpy as np
import argparse

def draw_grid(frame, start_x, end_x, start_y, end_y, x_step, y_step):
    h, w, _ = frame.shape
    grid_img = frame.copy()

    def x_to_px(x):
        return int((x - start_x) / (end_x - start_x) * w)

    def y_to_px(y):
        return int((y - start_y) / (end_y - start_y) * h)

    for x in range(start_x, end_x + 1, x_step):
        px = x_to_px(x)
        cv2.line(grid_img, (px, 0), (px, h), (0, 255, 0), 1)
        cv2.putText(grid_img, str(x), (px + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    for y in range(start_y, end_y + 1, y_step):
        py = y_to_px(y)
        cv2.line(grid_img, (0, py), (w, py), (0, 255, 0), 1)
        cv2.putText(grid_img, str(y), (5, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return grid_img

def main():
    parser = argparse.ArgumentParser(description="Overlay a customizable grid on a video.")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save output video (optional)")
    parser.add_argument("--start_x", type=int, default=-90, help="Start value for X axis")
    parser.add_argument("--end_x", type=int, default=90, help="End value for X axis")
    parser.add_argument("--start_y", type=int, default=0, help="Start value for Y axis")
    parser.add_argument("--end_y", type=int, default=140, help="End value for Y axis")
    parser.add_argument("--x_step", type=int, default=10, help="Step size for X grid lines")
    parser.add_argument("--y_step", type=int, default=10, help="Step size for Y grid lines")

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if args.output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_path, fourcc, fps, (frame_w, frame_h))
    else:
        out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_grid = draw_grid(
            frame,
            args.start_x,
            args.end_x,
            args.start_y,
            args.end_y,
            args.x_step,
            args.y_step
        )

        cv2.imshow("Grid Overlay", frame_with_grid)
        if out:
            out.write(frame_with_grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
