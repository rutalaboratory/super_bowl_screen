import cv2
import numpy as np


def draw_origin(image, origin, size):
    x, y = origin
    length = size

    # Draw axes: X (red), Y (green), Z (blue)
    x_axis = (int(x + length), y)
    y_axis = (int(x + length * 0.7), int(y + length * 0.7))
    z_axis = (x, int(y - length))

    image = cv2.arrowedLine(image, (x, y), x_axis, (0, 0, 255), 2, tipLength=0.2)
    image = cv2.putText(image, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    image = cv2.arrowedLine(image, (x, y), y_axis, (0, 255, 0), 2, tipLength=0.2)
    image = cv2.putText(image, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    image = cv2.arrowedLine(image, (x, y), z_axis, (255, 0, 0), 2, tipLength=0.2)
    image = cv2.putText(image, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return image

def resize_to_fit_screen(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    return cv2.resize(image, (int(width * scale), int(height * scale)))

def main():
    image_path = '/home/murtaza/Downloads/IMG_3665.jpg'  # Replace with your image path
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to load image.")
        return

    image = resize_to_fit_screen(image)
    clone = image.copy()
    size = 50  # Default axis length
    origin = None

    def click_event(event, x, y, flags, param):
        nonlocal origin, image
        if event == cv2.EVENT_LBUTTONDOWN:
            origin = (x, y)
            image = draw_origin(clone.copy(), origin, size)
            cv2.imshow('Image with Origin', image)

    cv2.namedWindow('Image with Origin')
    cv2.setMouseCallback('Image with Origin', click_event)
    cv2.imshow('Image with Origin', image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and origin is not None:
            cv2.imwrite('image_with_origin.jpg', image)
            print('Saved as image_with_origin.jpg')
        elif key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()