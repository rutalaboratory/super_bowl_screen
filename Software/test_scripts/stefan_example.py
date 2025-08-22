import numpy as np
import jax.numpy as jnp
import pylab as plt


def rotate(src, spher_coords, roll, pitch, yaw, width, height):
    pi = jnp.pi
    roll = jnp.deg2rad(roll)
    pitch = jnp.deg2rad(pitch)
    yaw = jnp.deg2rad(yaw)

    # Rotation matrices
    R_x = jnp.array([[1, 0, 0],
                     [0, jnp.cos(roll), -jnp.sin(roll)],
                     [0, jnp.sin(roll), jnp.cos(roll)]])
    R_y = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)],
                     [0, 1, 0],
                     [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    R_z = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],
                     [jnp.sin(yaw), jnp.cos(yaw), 0],
                     [0, 0, 1]])
    
    # Combined rotation
    R = jnp.dot(jnp.dot(R_x, R_y), R_z)

    # Rotate the spherical coordinates
    spherical_coords = jnp.moveaxis(spher_coords, 0, -1).reshape(-1, 3)
    spherical_coords = jnp.dot(spherical_coords, R)
    spherical_coords = spherical_coords.reshape(*spher_coords[0].shape, 3)
    
    # Convert back to equirectangular coordinates
    lng = jnp.arctan2(spherical_coords[:, :, 1], spherical_coords[:, :, 0])
    lat = jnp.arctan2(spherical_coords[:, :, 2],
                      jnp.sqrt(spherical_coords[:, :, 0]**2 + spherical_coords[:, :, 1]**2))

    ix = (0.5 * lng / pi + 0.5) * width - 0.5
    iy = (lat / pi + 0.5) * height - 0.5

    # Safely clip the indices to avoid out-of-bounds errors
    ix_ = jnp.clip(jnp.round(ix % width), 0, width - 1).astype(int)
    iy_ = jnp.clip(jnp.round(iy), 0, height - 1).astype(int)

    # Sample pixels from the rotated coordinates
    dest = src[iy_, ix_, :]
    return dest


class Stimulus():
    def __init__(self, img_size, fov_azi=0, fov_ele=0):
        self.width = img_size[1]
        self.height = img_size[0]
        self.fov_azi = fov_azi
        self.fov_ele = fov_ele
        pi = jnp.pi

        # Equirectangular pixel coordinates
        x, y = jnp.meshgrid(jnp.arange(self.width), jnp.arange(self.height))
        xx = 2 * (x + 0.5) / self.width - 1
        yy = 2 * (y + 0.5) / self.height - 1
        lng = pi * xx
        lat = 0.5 * pi * yy

        # Convert to spherical 3D coordinates
        X = jnp.cos(lat) * jnp.cos(lng)
        Y = jnp.cos(lat) * jnp.sin(lng)
        Z = jnp.sin(lat)

        self.spher_coords = jnp.array([X, Y, Z])

    def rot_equi_img(self, src, roll, pitch, yaw):
        return rotate(src, self.spher_coords, roll, pitch, yaw, self.width, self.height)


# === Test with RGB image ===
img_gray = np.zeros((360, 720), dtype="uint8")
img_gray[0:10, :] = 255  # White line at the top

# Convert grayscale to RGB
img_rgb = np.stack([img_gray]*3, axis=-1)  # shape: (360, 720, 3)

img_rgb[0:280,180,1]= 255
img_rgb[0:280,540,1]= 255
img_rgb[280,180:540,1]= 255

Stim = Stimulus((360, 720), 360, 180)

img_rot = Stim.rot_equi_img(img_rgb, roll=0, pitch=-110, yaw=0)

img_rot2 = Stim.rot_equi_img(img_rot, roll=0, pitch=0, yaw=-90)

img_rot_3 = Stim.rot_equi_img(img_rot, roll=0, pitch=-20, yaw=0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(img_rgb)

plt.subplot(1, 4, 2)
plt.title("Rotated X -90Â°")
plt.imshow(np.array(img_rot, dtype=np.uint8))

plt.subplot(1, 4, 3)
plt.title("Rotated Y 45Â°")
plt.imshow(np.array(img_rot2, dtype=np.uint8))

plt.subplot(1, 4, 4)
plt.title("Rotated Z 45Â°")
plt.imshow(np.array(img_rot_3, dtype=np.uint8))

plt.tight_layout()
plt.show()


