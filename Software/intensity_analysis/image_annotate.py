import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import sys
import yaml
import os

CROP_FILE = "crop_params.yml"

def save_crop_params(params):
    with open(CROP_FILE, 'w') as f:
        yaml.dump({'x': params[0], 'y': params[1], 'w': params[2], 'h': params[3]}, f)

def load_crop_params():
    if os.path.exists(CROP_FILE):
        with open(CROP_FILE, 'r') as f:
            data = yaml.safe_load(f)
            return (data['x'], data['y'], data['w'], data['h'])
    return None

def select_images():
    return filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])

def crop_image(image, existing_crop=None):
    # Get screen size
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    img_height, img_width = image.shape[:2]
    scale = min(screen_width / img_width, screen_height / img_height, 1.0)
    resized_image = image
    if scale < 1.0:
        resized_image = cv2.resize(image, (int(img_width * scale), int(img_height * scale)), interpolation=cv2.INTER_AREA)

    if existing_crop:
        x, y, w, h = existing_crop
        # Scale to resized image
        x_r = int(x * scale)
        y_r = int(y * scale)
        w_r = int(w * scale)
        h_r = int(h * scale)
        preview = resized_image[y_r:y_r+h_r, x_r:x_r+w_r].copy()
        cv2.rectangle(resized_image, (x_r, y_r), (x_r + w_r, y_r + h_r), (0, 255, 0), 2)
        cv2.imshow("Crop Preview - Press 'c' to change or any other key to keep", resized_image)
        key = cv2.waitKey(0)
        cv2.destroyWindow("Crop Preview - Press 'c' to change or any other key to keep")
        if key != ord('c'):
            return image[y:y+h, x:x+w], (x, y, w, h)

    roi = cv2.selectROI("Crop Image (ESC to skip)", resized_image, False, False)
    cv2.destroyWindow("Crop Image (ESC to skip)")

    if roi == (0, 0, 0, 0):
        return image, None

    x, y, w, h = roi
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)
    return image[y:y+h, x:x+w], (x, y, w, h)



def add_caption(image, caption_text, font_scale=2, thickness=4, padding=20):
    if not caption_text:
        return image

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(caption_text, font, font_scale, thickness)
    text_width, text_height = text_size

    h, w = image.shape[:2]
    caption_height = text_height + padding * 2
    new_image = np.full((h + caption_height, w, 3), 255, dtype=np.uint8)
    new_image[:h] = image

    text_x = (w - text_width) // 2
    text_y = h + padding + text_height

    cv2.putText(new_image, caption_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return new_image

def resize_to_same_height(images, target_height=None):
    if target_height is None:
        target_height = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_width = int((target_height / h) * w)
        resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
        resized.append(resized_img)
    return resized

def show_preview(image, root):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    preview = tk.Toplevel(root)
    preview.title("Stitched Image Preview")

    max_size = (1000, 800)
    img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)

    img_tk = ImageTk.PhotoImage(img_pil)
    label = tk.Label(preview, image=img_tk)
    label.image = img_tk
    label.pack()

    def on_save():
        save_image(image)
        preview.destroy()
        root.quit()

    save_btn = tk.Button(preview, text="Save Image", command=on_save)
    save_btn.pack(pady=10)

def save_image(image):
    filepath = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
    )
    if not filepath:
        messagebox.showwarning("Cancelled", "Save cancelled.")
        return

    if not isinstance(image, np.ndarray) or image.ndim != 3:
        messagebox.showerror("Error", "Image data is invalid or not in expected format.")
        return

    success = cv2.imwrite(filepath, image)
    if success:
        messagebox.showinfo("Saved", f"Image saved to: {filepath}")
    else:
        messagebox.showerror("Error", f"Failed to save image to: {filepath}")

def main():
    root = tk.Tk()
    root.withdraw()

    paths = select_images()
    if not paths:
        print("No images selected.")
        return

    use_same_crop = messagebox.askyesno("Crop Option", "Would you like to use the same crop area for all images?")
    crop_coords = None

    if use_same_crop:
        # Load saved crop if available
        previous_crop = None
        if os.path.exists(CROP_FILE):
            use_saved = messagebox.askyesno("Saved Crop Found", "Use previously saved crop parameters?")
            if use_saved:
                previous_crop = load_crop_params()

        # Load first image and allow crop/preview
        first_image = cv2.imread(paths[0])
        if first_image is None:
            messagebox.showerror("Error", f"Could not read {paths[0]}")
            return

        cropped_image, crop_coords = crop_image(first_image, existing_crop=previous_crop)
        if crop_coords:
            save_crop_params(crop_coords)
    else:
        crop_coords = None

    images_with_captions = []
    for idx, path in enumerate(paths):
        image = cv2.imread(path)
        if image is None:
            continue

        if use_same_crop and crop_coords:
            x, y, w, h = crop_coords
            cropped_image = image[y:y+h, x:x+w]
        else:
            cropped_image, _ = crop_image(image)

        caption = simpledialog.askstring("Caption", f"Enter caption for {path} (optional):")
        image_with_caption = add_caption(cropped_image, caption)
        image_with_caption = image_with_caption.astype(np.uint8)
        images_with_captions.append(image_with_caption)

    if not images_with_captions:
        print("No valid images loaded.")
        return

    resized_images = resize_to_same_height(images_with_captions)
    stitched = cv2.hconcat(resized_images)
    show_preview(stitched, root)

    root.mainloop()
    sys.exit()


if __name__ == "__main__":
    main()
