import os
import cv2
import numpy as np
import threading
from scipy.spatial import cKDTree
from pypylon import pylon  # Basler SDK

# --- Step 1: Generate projector image ---
def project_pixel_indices(res=(1280, 720), grid_size=(6, 6)):
    img = np.zeros((res[1], res[0], 3), dtype=np.uint8)
    coords = []

    cols, rows = grid_size
    x_spacing = res[0] // (cols + 1)
    y_spacing = res[1] // (rows + 1)

    for i in range(rows):
        for j in range(cols):
            x = (j + 1) * x_spacing
            y = (i + 1) * y_spacing
            cv2.circle(img, (x, y), 8, (255, 255, 255), -1)
            cv2.putText(img, f"{x},{y}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
            coords.append((x, y))

    return img, coords

# --- Step 2: Projector window thread ---
def show_projector_image(image, stop_event):
    cv2.namedWindow("Projector", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Projector", 1920, 0)  # Adjust for your projector position
    cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while not stop_event.is_set():
        cv2.imshow("Projector", image)
        if cv2.waitKey(1) == 27:
            stop_event.set()
            break

    cv2.destroyWindow("Projector")

# --- Step 3: Capture from Basler camera ---
def capture_basler_image(output_path="basler_capture.png"):
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbingMax(1)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab.GrabSucceeded():
            img = converter.Convert(grab)
            img = img.GetArray()
            cv2.imwrite(output_path, img)
            camera.Close()
            print("[INFO] Image captured")
            return output_path
    camera.Close()
    raise RuntimeError("Failed to capture image from Basler.")

# --- Step 4: Manual point selection with cache ---
def pick_points_with_cache_from_current_image(latest_image, save_path="picked_points.npy"):
    scale_factor = 0.8 * min(1920 / latest_image.shape[1], 1080 / latest_image.shape[0])
    resized = cv2.resize(latest_image, (0, 0), fx=scale_factor, fy=scale_factor)
    clone = resized.copy()

    if os.path.exists(save_path):
        loaded_points = np.load(save_path)
        show_img = clone.copy()
        for pt in loaded_points:
            pt_scaled = (int(pt[0] * scale_factor), int(pt[1] * scale_factor))
            cv2.circle(show_img, pt_scaled, 5, (255, 0, 0), -1)
        cv2.putText(show_img, "Press 'y' to reuse, any other key to reselect",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Reuse Points?", show_img)
        key = cv2.waitKey(0)
        cv2.destroyWindow("Reuse Points?")
        if key in [ord('y'), ord('Y')]:
            return loaded_points.tolist()

    clicked = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and clicked:
            clicked.pop()

    cv2.namedWindow("Click Points")
    cv2.setMouseCallback("Click Points", click)

    print("Left-click to select points; Right-click to undo; Press ENTER to finish.")

    while True:
        disp = clone.copy()
        for pt in clicked:
            cv2.circle(disp, pt, 5, (0, 0, 255), -1)
        cv2.imshow("Click Points", disp)

        key = cv2.waitKey(10)
        if key == 13 or key == 10:
            break

    cv2.destroyWindow("Click Points")

    unscaled = [(pt[0] / scale_factor, pt[1] / scale_factor) for pt in clicked]
    np.save(save_path, np.array(unscaled))

    return unscaled

# --- Step 5: Calibration ---
def calibrate_projector_from_manual_clicks(projector_coords, camera_coords, image_shape):
    if len(camera_coords) == 0:
        raise ValueError("No points selected.")

    projector_coords = np.array(projector_coords)
    camera_coords = np.array(camera_coords)

    tree = cKDTree(projector_coords)

    matched_projector_coords = []
    used_indices = set()

    for cam_pt in camera_coords:
        dist, idx = tree.query(cam_pt)
        while idx in used_indices:
            projector_coords[idx] = np.array([np.inf, np.inf])
            tree = cKDTree(projector_coords)
            dist, idx = tree.query(cam_pt)

        matched_projector_coords.append(projector_coords[idx])
        used_indices.add(idx)

    objpoints = np.array([[x, y, 0] for x, y in matched_projector_coords], dtype=np.float32).reshape(-1, 1, 3)
    imgpoints = np.array(camera_coords, dtype=np.float32).reshape(-1, 1, 2)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objpoints], [imgpoints], image_shape, None, None)

    print("Calibration RMS error:", ret)
    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", dist.ravel())
    print("Rotation vector:\n", rvecs[0].ravel())
    print("Translation vector:\n", tvecs[0].ravel())

    return K, dist, rvecs[0], tvecs[0]

def reproject_camera_to_projector(camera_points, K, dist, rvec, tvec):
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec

    # Undistort and normalize the camera points
    camera_points = np.array(camera_points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(camera_points, K, dist, P=K)
    undistorted = undistorted.reshape(-1, 2)

    # Convert each point to projector space assuming Z = 0 (screen is flat in projector frame)
    projected_points = []
    for pt in undistorted:
        x_c, y_c = pt
        cam_point = np.array([x_c, y_c, 1.0])  # In normalized camera coordinates

        # Backproject to projector space (Z=0 plane)
        ray = R_inv @ cam_point
        if abs(ray[2]) < 1e-6:
            print(f"[WARN] Skipping point due to ray.z ≈ 0: ray = {ray}")
            continue

        scale = -t_inv[2] / ray[2]
        world_point = R_inv @ cam_point * scale + t_inv

        # world_point should correspond to projector pixel (x, y)
        projected_points.append((world_point[0], world_point[1]))

    return projected_points
  

# --- Main script ---
def main():
    projector_res = (1280, 720)
    cam_res = (1280, 960)

    proj_img, projector_coords = project_pixel_indices(projector_res)

    stop_event = threading.Event()
    projector_thread = threading.Thread(target=show_projector_image, args=(proj_img, stop_event))
    projector_thread.start()

    input("[ACTION] Press Enter to capture image from Basler camera...")
    image_path = capture_basler_image()

    stop_event.set()
    projector_thread.join()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    camera_coords = pick_points_with_cache_from_current_image(img)

    K, dist, rvecs, tvecs = calibrate_projector_from_manual_clicks(projector_coords, camera_coords, cam_res)

    projected_back = reproject_camera_to_projector(camera_coords, K, dist, rvecs, tvecs)

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    for pt in projected_back:
       print("DEBUG", pt[0])
       x, y = int(np.round(pt[0].item())), int(np.round(pt[1].item()))
       if 0 <= x < 1280 and 0 <= y < 720:
            cv2.circle(canvas, (x, y), 8, (0, 255, 0), -1)

    cv2.namedWindow("Reprojected Dots", cv2.WINDOW_NORMAL)
    cv2.imshow("Reprojected Dots", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
