import numpy as np
import cv2
import jax.numpy as jnp
import jax 
from jax import jit , lax

import pylab as plt


@jit
def rotate(src, dest, spher_coords, roll, pitch, yaw, width, height):
    """
    Rotates a spherically projected image by applying 3D rotation matrices.
    
    Parameters:
    - src: Source image array to rotate
    - dest: Destination array for rotated image
    - spher_coords: Pre-computed 3D spherical coordinates for each pixel
    - roll: Rotation around X-axis in degrees (tilt)
    - pitch: Rotation around Y-axis in degrees (up/down)
    - yaw: Rotation around Z-axis in degrees (left/right)
    - width: Image width in pixels
    - height: Image height in pixels
    
    Returns:
    - dest: Rotated image array
    """
    pi = jnp.pi
    # Convert rotation angles from degrees to radians
    roll = jnp.deg2rad(roll)
    pitch = jnp.deg2rad(pitch)
    yaw = jnp.deg2rad(yaw)
    
    # Create 3D rotation matrices for each axis
    # R_x: rotation around X axis (roll)
    R_x = jnp.array([[1, 0, 0], [0, jnp.cos(roll), -jnp.sin(roll)], [0, jnp.sin(roll), jnp.cos(roll)]])
    # R_y: rotation around Y axis (pitch)
    R_y = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    # R_z: rotation around Z axis (yaw)
    R_z = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0], [jnp.sin(yaw), jnp.cos(yaw), 0], [0, 0, 1]])
    # Combine rotations: R = R_x * R_y * R_z
    R = jnp.dot(jnp.dot(R_x, R_y), R_z)
    
    # Prepare spherical coordinates for rotation
    # Move axis to make the array shape compatible with matrix multiplication
    spherical_coords = jnp.moveaxis(spher_coords, 0, -1)
    # Reshape to 2D array where each row is a 3D point
    spherical_coords = spherical_coords.reshape(-1, 3)
    # Apply rotation to all points at once
    spherical_coords = jnp.dot(spherical_coords, R)
    # Reshape back to original dimensions
    spherical_coords = spherical_coords.reshape(*spher_coords[0].shape, 3)
    
    # Convert rotated 3D coordinates back to spherical coordinates
    # Calculate longitude (lng) and latitude (lat) from 3D coordinates
    lng = jnp.arctan2(spherical_coords[:,:,1], spherical_coords[:,:,0])
    lat = jnp.arctan2(spherical_coords[:,:,2], jnp.sqrt(spherical_coords[:,:,0]**2 + spherical_coords[:,:,1]**2))

    # Convert spherical coordinates to image coordinates
    # Map longitude [-π, π] to [0, width] and latitude [-π/2, π/2] to [0, height]
    ix = (0.5 * lng / pi + 0.5) * width - 0.5
    iy = (lat / pi + 0.5) * height - 0.5
    # Sample the source image at calculated coordinates
    # Round coordinates to nearest pixel and handle wrapping around in x-direction
    dest = src[jnp.round(iy).astype(int), jnp.round((ix) % width).astype(int),:]
    return dest
    

class Stimulus():
    """
    The Stimulus class handles the creation and transformation of visual stimuli for the bowl arena.
    It converts flat images into the correct format for spherical projection, handling all the
    mathematical transformations needed to make the image look correct when projected onto a curved surface.
    """
    
    def __init__(self,img_size, fov_azi=0, fov_ele=0):
        """
        Initialize the stimulus generator.
        
        Parameters:
        - img_size: Tuple of (height, width, channels) for the stimulus image
        - fov_azi: Field of view in azimuth (horizontal) direction in degrees
        - fov_ele: Field of view in elevation (vertical) direction in degrees
        """
        # Store basic dimensions for the projection
        self.width = img_size[1]    # Image width in pixels
        self.height = img_size[0]   # Image height in pixels
        self.fov_azi = fov_azi      # Horizontal field of view
        self.fov_ele = fov_ele      # Vertical field of view
        pi = jnp.pi                 
        
        # Create coordinate grid for the image
        # This creates two 2D arrays where each point contains its x,y coordinates
        x, y = jnp.meshgrid(jnp.arange(self.width), jnp.arange(self.height))

        # Convert pixel coordinates to normalized coordinates (-1 to 1)
        # Adding 0.5 centers the coordinates in each pixel
        xx = 2 * (x + 0.5) / self.width - 1    # Normalize x coordinates
        yy = 2 * (y + 0.5) / self.height - 1   # Normalize y coordinates

        # Convert normalized coordinates to spherical coordinates
        # lng (longitude) ranges from -π to π
        # lat (latitude) ranges from -π/2 to π/2
        lng = pi * xx    # Convert x to longitude angles
        lat = 0.5 * pi * yy  # Convert y to latitude angles
    
        # Convert spherical coordinates to 3D Cartesian coordinates
        # This maps each pixel to a point on a unit sphere
        X = jnp.cos(lat) * jnp.cos(lng)  # x = r * cos(lat) * cos(lng)
        Y = jnp.cos(lat) * jnp.sin(lng)  # y = r * cos(lat) * sin(lng)
        Z = jnp.sin(lat)                 # z = r * sin(lat)
        
        # Store the 3D coordinates for later use in rotation
        # Each pixel now has an (x,y,z) coordinate on the unit sphere
        self.spher_coords = jnp.array([X, Y, Z])
        
    def rot_equi_img(self,src, dest, roll, pitch, yaw):
        """
        Rotate the equirectangular image by the specified angles.
        
        Parameters:
        - src: Source image to rotate
        - dest: Destination buffer for the rotated image
        - roll: Rotation around the x-axis (degrees)
        - pitch: Rotation around the y-axis (degrees)
        - yaw: Rotation around the z-axis (degrees)
        
        Returns:
        - Rotated image in the destination buffer
        """
        # Pass the stored spherical coordinates and rotation angles to the @jit optimized rotate function
        return rotate(src,dest,self.spher_coords,roll,pitch,yaw,self.width,self.height)


def projection(frame,rhos,phis):
    return frame[rhos,phis,:]

projection_jit = jax.jit(projection)

@jit
def select_fov(image):
        # return image[0:840, 0:1080,:]
        return image[0:280,180:540,:]
    
@jit
def write_fov(image,insertion):
        return image.at[0:280,180:540,:].set(insertion)


@jit
def apply_mask(img, mask):
    return img * jnp.bitwise_and(mask[..., jnp.newaxis], 1)
        

@jit
def insert_image(large_img, small_img, position):
    y, x = position
    h, w = small_img.shape[:2]
    mask = jnp.zeros_like(large_img)
    mask = lax.dynamic_update_slice(mask, small_img, (y, x, 0))
    return mask#jnp.where(mask == 0, large_img, mask)


class Projector:
    """
    A class to handle the projection of images onto a spherical surface display.
    
    This class manages the transformation of flat images into a format suitable for
    projection onto a curved surface, handling aspects such as field of view,
    masking, and coordinate transformation.
    """
    
    def __init__(self, res_x=1280, res_y=720, proj_x=1280, proj_y=640, fov_azi=(0, 180), fov_ele=(15, 140)):
        """
        Initialize the projector with display and projection parameters.
        
        Parameters:
        -----------
        res_x : int, optional (default=1280)
            Total horizontal resolution of the display
        res_y : int, optional (default=720)
            Total vertical resolution of the display
        proj_x : int, optional (default=1280)
            Width of the projected area
        proj_y : int, optional (default=640)
            Height of the projected area
        """
        # Default stimulus dimensions and field of view
        self.stim_x = 360 
        self.stim_y = 180  
        self.fov_azi = fov_azi  # Azimuth field of view range (degrees)
        self.fov_ele = fov_ele  # Elevation field of view range (degrees)
        
        # Display parameters
        self.resolution = (res_x, res_y)  # Total display resolution
        self.projected_area = (proj_x, proj_y)  # Active projection area
        
        # Initialize blank screens for rendering
        self.blank_screen = np.zeros([self.resolution[1], self.resolution[0], 3], dtype="uint8")
        self.mask_screen = jnp.zeros([self.resolution[1], self.resolution[0], 3], dtype="uint8")
        
        # Calculate border width for centering the projection
        self.border = int((self.resolution[0] - self.projected_area[0]) / 2)
        
    def initialize_projection_matrix(self, stim_size, fov_azi, fov_ele):
        """
        Initialize the projection transformation matrices and masks.
        
        This method creates the coordinate mappings needed to transform
        a flat image into the curved projection space.
        
        Parameters:
        -----------
        stim_size : tuple
            Size of the stimulus (height, width)
        fov_azi : tuple
            Azimuth field of view range (min, max) in degrees
        fov_ele : tuple
            Elevation field of view range (min, max) in degrees
        """
        # Update stimulus dimensions and field of view
        self.stim_x = stim_size[1]
        self.stim_y = stim_size[0]
        print("projector, stim_size: ", stim_size)
        self.fov_azi = fov_azi
        self.fov_ele = fov_ele
        
        # Set up projection dimensions
        xdim = self.projected_area[0]
        ydim = self.projected_area[1]
        xcenter = int(self.projected_area[0] / 2)
        positiony = 0
        
        # Create coordinate matrices
        x_ones = np.ones(xdim)
        y_ones = np.ones(ydim).T
        x_vec = np.linspace(1, xdim, xdim)
        y_vec = np.linspace(1, ydim, ydim).T
        
        # Generate coordinate grids
        ymat = np.outer(y_vec, x_ones)
        xmat = np.outer(y_ones, x_vec)
        
        # Calculate polar coordinates
        # rhos: radial distance from center (elevation angle)
        rhos = (np.around((np.sqrt((xmat - xcenter)**2 + 
                                  (ymat - positiony)**2)) / xcenter * self.stim_y)).astype(int)
        # phis: angular position (azimuth angle)
        phis = (np.around((np.arctan2((ymat - positiony), 
                                     (xmat - xcenter))) / np.pi * self.stim_x)).astype(int)
        
        # Create and apply projection mask
        mask = np.zeros([ydim, xdim])
        inner = self.stim_y / self.fov_ele[1] * self.fov_ele[0]
        mask[np.where((rhos <= self.stim_y) & (rhos >= inner))] = 255
        self.mask = np.asarray(mask, dtype="uint8")
        
        # Store clipped coordinate matrices for projection
        self.phis = np.clip(phis, 0, self.stim_x)
        self.rhos = np.clip(rhos, 0, self.stim_y)
    
    def project_image(self, image):
        """
        Project an image using the pre-computed transformation matrices.
        
        Parameters:
        -----------
        image : ndarray
            The source image to project
            
        Returns:
        --------
        ndarray
            The projected image
        """
        return projection_jit(image, self.rhos, self.phis)
    
    def mask_image(self, image):
        """
        Apply masking to the projected image and insert it into the display frame.
        
        Parameters:
        -----------
        image : ndarray
            The image to mask and insert
            
        Returns:
        --------
        ndarray
            The final display frame with the masked image inserted
        """
        projektor = apply_mask(image, self.mask)
        self.mask_screen = insert_image(self.mask_screen, projektor, (0, self.border))
        return np.asarray(self.mask_screen, dtype="uint8") 