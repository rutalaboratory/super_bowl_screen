import rawpy
import numpy as np
import matplotlib.pyplot as plt

# Ask user for DNG file path
dng_path = input("Enter path to DNG file: ")

# Load raw image without postprocessing
with rawpy.imread(dng_path) as raw:
    raw_image = raw.raw_image_visible.astype(np.uint16)

# Display the raw image (as grayscale)
plt.imshow(raw_image, cmap='gray')
plt.title('Raw Image (Unprocessed Bayer Data)')
plt.axis('off')
plt.show()
