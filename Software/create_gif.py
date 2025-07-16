import imageio
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont

def create_gif():
    # Open a file dialog to select images
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_paths = filedialog.askopenfilenames(
        title="Select images to create GIF",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("JPEG files", "*.jpeg"),
            ("Bitmap files", "*.bmp"),
            ("TIFF files", "*.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not file_paths:
        print("No images selected.")
        return

    # Ask for the output file name
    output_path = filedialog.asksaveasfilename(
        title="Save GIF as...",
        defaultextension=".gif",
        filetypes=[("GIF files", "*.gif")]
    )

    if not output_path:
        print("No output file selected.")
        return

    # Values to write on each frame
    frame_texts = [1023, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

    if len(file_paths) != len(frame_texts):
        print(f"Error: You selected {len(file_paths)} images but {len(frame_texts)} texts are expected.")
        return

    # Create GIF
    images = []
    for path, text in zip(file_paths, frame_texts):
        img = Image.open(path).convert('RGB')

        # Resize image to 20% of original size
        img = img.resize((int(img.width * 0.2), int(img.height * 0.2)), Image.LANCZOS)

        draw = ImageDraw.Draw(img)

        # You can choose a font; fallback if no font available
        try:
            font = ImageFont.truetype("arial.ttf", 128)  # Smaller font after resizing
        except IOError:
            font = ImageFont.load_default()

        # Determine text size and position
        text_str = str(text)
        bbox = draw.textbbox((0, 0), text_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        img_width, img_height = img.size
        x = (img_width - text_width) / 2
        y = img_height - text_height - 50  # 5 pixels from the bottom

        # Draw text
        draw.text((x, y), text_str, font=font, fill=(255, 255, 255))

        # Append frame
        images.append(img)

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=300,  # 0.5 seconds per frame
        loop=0
    )

    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    create_gif()
