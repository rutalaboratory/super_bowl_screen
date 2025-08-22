import numpy as np
import matplotlib.pyplot as plt
from bowl import *
import time
import sys
import functools



class Stimulation_Pipeline():
    """
    The Stimulation_Pipeline class manages the display of visual stimuli in a on the spherical screen.
    It handles the conversion from regular images to the distorted projections needed for the bowl shape,
    and manages the display window positioning and timing.
    """
    
    def __init__(self,img_size=(360, 720,3), fov_azi=(0,180), fov_ele=(15,140),img_offsetx=3840+3240,img_offsety=2400,name = "Arena", projector_width_pixels=1280, debug=False):
        """
        Initialize the stimulus projection system.

        Parameters:
        - img_size: Size of the source image (height, width, channels). Default is 360x720 pixels with 3 color channels
        - fov_azi: Field of view in azimuth (horizontal) direction in degrees. Default is 0 to 180 degrees
        - fov_ele: Field of view in elevation (vertical) direction in degrees. Default is 15 to 140 degrees
        - img_offsetx, img_offsety: Position of the projection window on the screen in pixels
        - name: Name of the display window
        - projector_width_pixels: Width of the projector resolution
        - debug: Enable debug output
        """
        self.debug = debug
        
        # Calculate how many pixels we need for the given field of view
        # For example, if we want to show 180 degrees in 720 pixels, each degree needs 4 pixels
        azi_pix = int(img_size[1]/360*fov_azi[1])  # Pixels needed for azimuth (horizontal) view
        ele_pix = int(img_size[0]/180*fov_ele[1])  # Pixels needed for elevation (vertical) view
    
        # Store pixel dimensions for later use
        self.azi_pix = azi_pix  # Width in pixels
        self.ele_pix = ele_pix  # Height in pixels
        
        # Store image dimensions and calculate resolution (degrees per pixel)
        self.xdim = img_size[1]  # Image width
        self.ydim = img_size[0]  # Image height
        self.resolution = np.array([1/(self.ele_pix/fov_ele[1]),1/(self.azi_pix/fov_azi[1])])  # Degrees per pixel
        self.image_size = img_size  # Full image size including color channels
        
        # Create a blank image buffer for stimulus generation
        self.dest = np.zeros(img_size,dtype = "uint8")
        
        # Initialize the components that handle stimulus generation and projection
        self.Stimulus = Stimulus(self.dest.shape)  # Handles stimulus creation and rotation
        
        # Set up projector parameters (this handles the bowl-shape distortion)
        self.Projector_1 = Projector()
        self.Projector_1.initialize_projection_matrix((ele_pix,azi_pix),fov_azi,fov_ele)
        
        # Initialize timing variables for stimulus presentation
        self.dt = 0          # Time elapsed since start
        self.time_start = 0  # Start time of stimulus
        self.frames = 0      # Number of frames shown
        self.oldframe = 0    # Previous frame (used for timing)
        

        # Set up the display window for the stimulus
        # This creates a window on the second monitor at the specified position

        self.WINDOW_NAME = name          # Name of the window
        self.width_first = img_offsetx   # X position of window (usually on second monitor)
        self.height_first = img_offsety  # Y position of window
        
        # Create and position the window
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.moveWindow(self.WINDOW_NAME, self.width_first, self.height_first)  # Move it to the right position
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Make it fullscreen
        
        # Print initialization info
        print("initialize Stimulation Pipeline < ",self.WINDOW_NAME," >: xdim=,",self.xdim,"ydim=",self.ydim,
              "at position x=", self.width_first,"y=", self.height_first)

    def project_dot_at(self, azimuth, elevation, radius=5, dot_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """
        Projects a single dot on the screen at the given azimuth and elevation (in degrees).
        Dot will be shown in dot_color on a background of bg_color.
        """

        # Validate bounds
        if not (self.Projector_1.fov_azi[0] <= azimuth <= self.Projector_1.fov_azi[1]):
            print("Azimuth out of bounds.")
            return
        if not (self.Projector_1.fov_ele[0] <= elevation <= self.Projector_1.fov_ele[1]):
            print("Elevation out of bounds.")
            return

        # Convert azimuth and elevation to pixel coordinates
        x = int((azimuth - self.Projector_1.fov_azi[0]) / 
                (self.Projector_1.fov_azi[1] - self.Projector_1.fov_azi[0]) * self.xdim)
        y = int((1 - (elevation - self.Projector_1.fov_ele[0]) /
                (self.Projector_1.fov_ele[1] - self.Projector_1.fov_ele[0])) * self.ydim)

        # Create background
        img = np.full((self.ydim, self.xdim, 3), bg_color, dtype=np.uint8)
        cv2.circle(img, (x, y), radius, dot_color, -1)

        # Project the image
        rotated = self.Stimulus.rot_equi_img(img, self.dest, 0, 0, 0)
        cropped = select_fov(rotated)
        masked = self.Projector_1.project_image(cropped)
        output = self.Projector_1.mask_image(masked)
        cv2.imshow(self.WINDOW_NAME, output)
        cv2.waitKey(10)


    def generate_grating_vertical(self,color1, color2,spatial_freq):
        
        xdim=self.xdim
        ydim=self.ydim
        pixperdeg = int(xdim/360*spatial_freq)
        pic = np.ones([ydim,xdim],dtype = "uint8")*color1
        for x in range(int(xdim/(pixperdeg))):
            pic[:,2*x*pixperdeg:(2*x+1)*pixperdeg] = color2
        return pic

    def generate_grating_horizontal(self,color1, color2,spatial_freq):
        xdim=self.xdim
        ydim=self.ydim
        pixperdeg = int(xdim/360*spatial_freq)
        pic = np.ones([ydim,xdim],dtype = "uint8")*color1
        for y in range(int(ydim/(pixperdeg))):
            pic[2*y*pixperdeg:(2*y+1)*pixperdeg,:] = color2
        return pic


    def generate_grating_checkerboard(self,color1,spatial_freq):
        xdim=self.xdim
        ydim=self.ydim
        pixperdeg = int(xdim/360*spatial_freq)
        pic = np.zeros([ydim,xdim],dtype = "uint8")
        pic1 = np.zeros([ydim,xdim],dtype = "uint8")
        for x in range(int(xdim/(pixperdeg))):
            pic[:,2*x*pixperdeg:(2*x+1)*pixperdeg] = 1
        for y in range(int(ydim/(pixperdeg))):
            pic1[2*y*pixperdeg:(2*y+1)*pixperdeg,:] = 1
        pic = ((pic+pic1)%2)*color1
        return pic                                                           

    def generate_grating_sine_vertical(self,amplitude, offset,spatial_freq):
        xdim=self.xdim
        ydim=self.ydim
        f = int(360/spatial_freq)
        t = np.linspace(0,2*np.pi*f,xdim)
        val = amplitude*np.sin(t)+offset
        ys = np.ones(ydim)
        pic = np.outer(ys,val)
        return pic.astype("uint8")

    def generate_grating_sine_horizontal(self,amplitude, offset,spatial_freq):
        xdim=self.xdim
        ydim=self.ydim
        f = int(360/spatial_freq)
        t = np.linspace(0,2*np.pi*f,ydim)
        val = amplitude*np.sin(t)+offset
        ys = np.ones(xdim)
        pic = np.outer(val,ys)
        return pic.astype("uint8")

    def generate_edge_vertical(self,color1, color2):
        xdim=self.xdim
        ydim=self.ydim
        pic = np.ones([ydim,xdim],dtype = "uint8")*color1
        pic[:,:int(xdim/2)]= color2
        return pic


    def show_dark_screen(self,duration):
        output2 = np.zeros((self.Projector_1.resolution[1], self.Projector_1.resolution[0],3),dtype = "uint8")
        cv2.imshow(self.WINDOW_NAME,output2)
        key = cv2.waitKey(int(duration*1000))


    def show_trigger(self):
        output2 = np.zeros((self.Projector_1.resolution[1], self.Projector_1.resolution[0],3),dtype = "uint8")
        output2[-45:-5,-45:-5] = 128
        cv2.imshow(self.WINDOW_NAME,output2)
        key = cv2.waitKey(3)

    #rotational execution

    def generate_rotational(self,texture,duration,roll=0,pitch=0,yaw=0,rot_offset=(0,30,0)):
        
        # the "generate_rotational" function creates a start time and manages runtime and timing it  uses an pre generated texture to project it onto the projector.
        # the pre generated texture can be rotated online in constant speed along every rotational axis.

        self.show_trigger()
        self.show_dark_screen(0.1)
        fpss = np.array([])
        dts = np.array([])
        fps = 0
        timer = cv2.getTickCount()

        Input_im = texture
        resized = cv2.cvtColor(Input_im, cv2.COLOR_GRAY2RGB)
        self.show_dark_screen(0.1)
        self.show_trigger()
        self.time_start = time.time()
        i=0
        while time.time() < self.time_start + duration:
            
            self.dt = time.time()-self.time_start
            rotated = self.Stimulus.rot_equi_img(resized,self.dest,roll*self.dt,pitch*self.dt,yaw*self.dt)
            rotated = self.Stimulus.rot_equi_img(rotated,self.dest,rot_offset[0],rot_offset[1],rot_offset[2])
            croped = select_fov(rotated)
            masked = self.Projector_1.project_image(croped)
            output = self.Projector_1.mask_image(masked)
            cv2.imshow(self.WINDOW_NAME,output)

            tick = cv2.getTickCount()-timer
            fps = cv2.getTickFrequency()/(tick)
            timer = cv2.getTickCount()
            fpss = np.append(fpss,fps)
            dts = np.append(dts,self.dt)

            key = cv2.waitKey(1)#pauses for 1ms seconds before fetching next image

            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
        self.show_trigger()
        self.show_dark_screen(0.1)
        # print (np.mean(fpss))
        #print ((dts))
        #plt.plot(dts)
        #linear = np.linspace(0,duration,len(dts))
        #plt.plot(linear,dts-linear)
     
        
    def generate(self,function,duration=0,rot_offset=(0,0,0),*args, **kwargs):
        
        # the "generate" function creates a start time and manages runtime and timing it  uses an online generated texture to project it onto the projector.
        # function is an input function or object, which generates online textures, which are then projected.
        # once generate is called in the main loop the insered function/or object is initialized. Afterwards the Code in the generate function is executed
        # in the While loop, the function is called and args and kwargs are transfered.
        
        self.show_trigger()
        self.show_dark_screen(0.1)
        fpss = np.array([])
        fps = 0
        timer = cv2.getTickCount()
        self.show_dark_screen(0.1)
        self.show_trigger()
        self.time_start = time.time()
        
        while time.time() < self.time_start + duration:
            Input_im = function(*args, **kwargs)
            
            if(len(Input_im.shape)<3):
                resized = cv2.cvtColor(Input_im, cv2.COLOR_GRAY2RGB)
            else:
                resized = Input_im
            if (rot_offset == (0,0,0)):
                rotated = resized
            else:
                rotated = self.Stimulus.rot_equi_img(resized,self.dest,rot_offset[0],rot_offset[1],rot_offset[2])
                
            croped =  (rotated)
            masked = self.Projector_1.project_image(croped)
            output = self.Projector_1.mask_image(masked)
            cv2.imshow(self.WINDOW_NAME,output)
            tick = cv2.getTickCount()-timer
            fps = cv2.getTickFrequency()/(tick)
            timer = cv2.getTickCount()
            fpss = np.append(fpss,fps)
            key = cv2.waitKey(1)#pauses for 1ms seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
        self.show_trigger()
        self.show_dark_screen(0.1)
        print(f"Average Loop FPS: {np.mean(fpss):.1f}, Total frames: {self.frames}")
        
        
    def looming_disk(self,radius,speed,distance,color_disc,color_bg,center=None):
        
        xdim=self.xdim
        ydim=self.ydim
        fac= int(ydim/180)
        if center is None: # use the middle of the image
            center = (int(xdim/2), int(ydim/2))
        else:
            center= np.asarray(center)
            center = center*fac
            
        self.dt = time.time()-self.time_start
        position = distance-(speed*self.dt)
        
        alpha = np.arctan((radius/position))
        pixel_radius = np.rad2deg(alpha)*fac
        
        Y, X = np.ogrid[:ydim, :xdim]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= pixel_radius
        pic = np.ones([ydim,xdim],dtype = "uint8")*color_bg
        pic[mask]=color_disc
        return pic
    
    def generate_dot(self, background_color, dot_color, dot_center, dot_radius):
        """
        Draw a dot on a grayscale background.

        Parameters:
            background_color (int): Grayscale value for the background (0–255).
            dot_color (int): Grayscale value for the dot (0–255).
            dot_center (tuple): (x, y) coordinates of the dot center in pixels.
            dot_radius (int): Radius of the dot in pixels.

        Returns:
            np.ndarray: Grayscale image with a dot.
        """
        xdim = self.xdim
        ydim = self.ydim
        img = np.ones((ydim, xdim), dtype=np.uint8) * background_color

        xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim))
        dist_squared = (xx - dot_center[0])**2 + (yy - dot_center[1])**2
        mask = dist_squared <= dot_radius**2
        img[mask] = dot_color

        return img


    

# each online calculated texture class consists of an initialization and an run function.
# this is necessary to initialize the stimulus parameters before runtime loop
# each online texture class is designed to get called by the Stimulation_Pipeline.generate() function

class ShowVideo():
    
    def __init__(self,arena,path,framerate=0,duration=0):
         
    
        self.arena = arena # object of the class Stimulation_Pipeline() 
        # This step is necessary in order to use functions and variables, such as time, from the stimulation pipeline class 
        # and to generate a separate initialisation function and a run function independently for each stimulus.
        
        self.video = cv2.VideoCapture(path)
        if framerate == 0:
            framerate = self.video.get(cv2.CAP_PROP_FPS)   
        frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frame count =",frame_count)
        if duration==0:
            duration = frame_count/framerate
        elif duration > frame_count/framerate:
            print("video file to short")
            duration = frame_count/framerate
        
        print("videoduration =",duration)
        self.arena.frames=0
        print("video framerate =",framerate)
        self.framerate = framerate
        fpss = np.array([])
        fps = 0
        timer = cv2.getTickCount()

    
  
    def run(self,):
                        
        theoretical_elapsed_time = self.arena.frames*(1/self.framerate)
        self.arena.dt = time.time()-self.arena.time_start 
        #print("dt = ",self.arena.dt,"  frames = ",self.arena.frames)
        
        if self.arena.dt>=theoretical_elapsed_time:
            
            ok, frame = self.video.read()#first frame ? 
            self.arena.frames += 1
            if not ok:
                print('Cannot read video file.')

        else:
            frame = self.arena.oldframe
        resized = cv2.resize(frame,self.arena.image_size[0:2], interpolation = cv2.INTER_AREA)
        self.arena.oldframe = frame
        return resized
    
# class MovingDotAzimuthal():
    
#     def __init__(self, arena, elevation=90, elevation_width=5, azi_limits=(30,150), speed=20):
#         self.arena = arena
#         self.elevation = elevation
#         self.elevation_width = elevation_width
#         self.azi_min, self.azi_max = azi_limits
#         self.speed = speed  # degrees/sec
#         self.direction = 1  # 1 for right, -1 for left

#         # calculate pixel radius based on elevation width
#         self.pixel_radius = int((elevation_width / 2) / self.arena.resolution[0])

#         # fixed pixel y position based on elevation
#         self.y = int((1 - (elevation - self.arena.Projector_1.fov_ele[0]) /
#                     (self.arena.Projector_1.fov_ele[1] - self.arena.Projector_1.fov_ele[0])) * self.arena.ydim)

#         # start at min azimuth
#         self.azi = self.azi_min
#         self.last_update_time = time.time()
#         self.pic = np.full((self.arena.ydim, self.arena.xdim, 3), 255, dtype=np.uint8)  # white background


#     def run(self):
#         current_time = time.time()
#         dt = current_time - self.last_update_time
#         self.last_update_time = current_time

#         # Update azimuth based on speed and direction
#         self.azi += self.speed * dt * self.direction

#         # Reflect direction if hitting bounds
#         if self.azi >= self.azi_max:
#             self.azi = self.azi_max
#             self.direction = -1
#         elif self.azi <= self.azi_min:
#             self.azi = self.azi_min
#             self.direction = 1

#         # Calculate x position in pixels
#         x = int((self.azi - self.arena.Projector_1.fov_azi[0]) /
#                 (self.arena.Projector_1.fov_azi[1] - self.arena.Projector_1.fov_azi[0]) * self.arena.xdim)

#         # Reset image
#         self.pic[:, :, :] = 255  # white background
#         cv2.circle(self.pic, (x, self.y), self.pixel_radius, (0, 0, 0), -1)

#         return self.pic

import numpy as np
import cv2
import time

import numpy as np
import cv2
import time

class MovingDotAzimuthal:
    def __init__(self, elevation_deg=90, elevation_width_deg=10, azi_limits=(30, 150), speed_dps=20, debug=False):
        # Image dimensions and field of view
        self.width = 720    # azimuth 0–360°
        self.height = 360   # elevation 0–180°
        self.elevation_min = 0
        self.elevation_max = 180
        self.azimuth_min = 0
        self.azimuth_max = 360

        self.elevation = elevation_deg
        self.elevation_width = elevation_width_deg
        self.azi_min, self.azi_max = azi_limits
        self.speed = speed_dps
        self.direction = 1
        self.debug = debug

        # Pixel radius for the dot
        self.pixel_radius = int((self.elevation_width / (self.elevation_max - self.elevation_min)) * self.height / 2)

        # Start azimuth
        self.azi = self.azi_min
        self.last_update_time = time.time()

        # Prepare image
        self.pic = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        if self.debug:
            self.draw_elevation_lines()

    def draw_elevation_lines(self):
        """
        Draw elevation lines with red color and label them with angle value.
        """
        line_color = (0, 0, 255)  # Red
        text_color = (0, 0, 255)  # Red
        pixel_per_degree = self.height / (self.elevation_max - self.elevation_min)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        label_x = 180  # fixed x position for labels

        for elevation in range(10, 181, 10):
            y = int(pixel_per_degree * elevation)
            if y >= self.height:
                continue

            # Draw horizontal line
            cv2.line(self.pic, (0, y), (self.width - 1, y), line_color, 1)

            # Label just above the line
            label = f"{elevation}"
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_y = max(y - 4, text_height + 2)  # avoid going off screen top

            cv2.putText(self.pic, label, (label_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def run(self):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update azimuth
        self.azi += self.speed * dt * self.direction
        if self.azi >= self.azi_max:
            self.azi = self.azi_max
            self.direction = -1
        elif self.azi <= self.azi_min:
            self.azi = self.azi_min
            self.direction = 1

        # Convert azimuth and elevation to pixel coordinates
        x = int((self.azi - self.azimuth_min) / (self.azimuth_max - self.azimuth_min) * self.width)
        y = int((self.elevation - self.elevation_min) / (self.elevation_max - self.elevation_min) * self.height)

        # Clear image
        self.pic[:, :, :] = 255  # white background
        if self.debug:
            self.draw_elevation_lines()

        # Draw black dot
        cv2.circle(self.pic, (x, y), self.pixel_radius, (0, 0, 0), -1)

        return self.pic


import numpy as np
import time
import cv2

class MovingDotElevation():

    def __init__(self, arena, azimuth=90, azimuth_width=5, ele_limits=(30,150), speed=20):
        self.arena = arena
        self.azimuth = azimuth
        self.azimuth_width = azimuth_width
        self.ele_min, self.ele_max = ele_limits
        self.speed = speed  # degrees/sec
        self.direction = 1  # 1 for up, -1 for down

        # calculate pixel radius based on azimuth width
        self.pixel_radius = int((azimuth_width / 2) / self.arena.resolution[1])

        # fixed pixel x position based on azimuth
        self.x = int((self.azimuth - self.arena.Projector_1.fov_azi[0]) /
                     (self.arena.Projector_1.fov_azi[1] - self.arena.Projector_1.fov_azi[0]) * self.arena.xdim)

        # start at min elevation
        self.ele = self.ele_min
        self.last_update_time = time.time()
        self.pic = np.full((self.arena.ydim, self.arena.xdim, 3), 255, dtype=np.uint8)  # white background

    def run(self):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update elevation based on speed and direction
        self.ele += self.speed * dt * self.direction

        # Reflect direction if hitting bounds
        if self.ele >= self.ele_max:
            self.ele = self.ele_max
            self.direction = -1
        elif self.ele <= self.ele_min:
            self.ele = self.ele_min
            self.direction = 1

        # Calculate y position in pixels
        y = int((1 - (self.ele - self.arena.Projector_1.fov_ele[0]) /
                 (self.arena.Projector_1.fov_ele[1] - self.arena.Projector_1.fov_ele[0])) * self.arena.ydim)

        # Reset image
        self.pic[:, :, :] = 255  # white background
        cv2.circle(self.pic, (self.x, y), self.pixel_radius, (0, 0, 0), -1)

        return self.pic


class LoomingDisk(): 
    
    def __init__(self,arena,center=None):
        self.arena = arena
        xdim=self.arena.xdim
        ydim=self.arena.ydim
        self.fac= int(ydim/180)
        if center is None: # use the middle of the image
            center = (int(xdim/2), int(ydim/2))
        else:
            self.center= np.asarray(center)
            self.center = self.center*self.fac
        self.pic = np.ones([ydim,xdim],dtype = "uint8")
        
    
    def run(self,radius,speed,distance,color_disc,color_bg):
        
     
        self.arena.dt = time.time()-self.arena.time_start
        position = distance-(speed*self.arena.dt)
        alpha = np.arctan((radius/position))
        pixel_radius = np.rad2deg(alpha)*self.fac
        Y, X = np.ogrid[:self.arena.ydim, :self.arena.xdim]
        dist_from_center = np.sqrt((X - self.center[0])**2 + (Y-self.center[1])**2)

        mask = dist_from_center <= pixel_radius
        self.pic = self.pic*color_bg
        self.pic[mask]=color_disc
        return self.pic
    

    
class ShowNoise():
    """
    This class creates a visual stimulus that displays random black and white noise 
    (like TV static) on the projection screen. The noise pattern updates at a specified 
    frame rate to create dynamic visual noise.
    """
    
    def __init__(self, arena, pixelsize, framerate=30, save_queue=None, debug=False):
        """
        Set up the noise stimulus with the given parameters.
        
        Args:
            arena: The projection environment where the stimulus will be shown
            pixelsize: Size of each noise pixel (larger numbers = bigger chunks of noise)
            framerate: How many times per second the noise pattern updates (default: 30Hz)
            save_queue: Optional multiprocessing queue to save generated patterns
        """
        # Store reference to the arena for accessing screen dimensions and timing
        self.debug = debug
        self.arena = arena
        self.save_queue = save_queue
        self.total_frames_generated = 0  # Track total frames generated
        xdim = self.arena.xdim
        ydim = self.arena.ydim
        
        # Create a blank black image to start with
        self.pic = np.zeros([ydim, xdim, 3], dtype="uint8")
        
        # Track frame saving
        self.saved_frames = 0
        
        # Calculate how many noise pixels we need based on the desired pixel size
        # These calculations ensure the noise scales properly across the whole arena
        self.y_noise = self.arena.ele_pix * self.arena.resolution[0] / pixelsize  # vertical noise pixels
        self.x_noise = self.arena.azi_pix * self.arena.resolution[1] / pixelsize  # horizontal noise pixels
        
        # Initialize frame counting and timing
        self.arena.frames = 0
        self.framerate = framerate
        print("video framerate =", framerate)
        print("y noise pixel = ", self.y_noise, " x noise pixel = ", self.x_noise)
        
        # Set random seed for reproducible noise patterns
        np.random.seed(0)
        timer = cv2.getTickCount()
        print("DEBUG", xdim, ydim, self.x_noise, self.y_noise)

    def run(self):
        """
        Generate and display the next frame of noise.
        This function is called repeatedly to create the animated noise effect.
        """
        # Calculate when the next frame should be shown based on the desired framerate
        theoretical_elapsed_time = self.arena.frames * (1/self.framerate)
        self.arena.dt = time.time() - self.arena.time_start
        
        # Print timing info every 60 frames
        if self.arena.frames % 60 == 0 and self.debug:
            timing_diff = self.arena.dt - theoretical_elapsed_time
            print(f"[Timing] Frame {self.arena.frames}: Real time={self.arena.dt:.3f}s, "
                  f"Theoretical={theoretical_elapsed_time:.3f}s, "
                  f"Diff={timing_diff*1000:.1f}ms")
        
        # If it's time for a new frame...
        if self.arena.dt >= theoretical_elapsed_time:
            # Create a new random noise pattern
            # First make small random black and white pixels
            Input_im = (np.random.randint(0, 2, (int(self.y_noise), int(self.x_noise), 1)) * 255).astype("uint8")
            
            # Convert the black and white image to RGB format
            image = cv2.cvtColor(Input_im, cv2.COLOR_GRAY2RGB)
            
            # Resize the noise to fill the entire screen
            resized = cv2.resize(image, dsize=(self.arena.xdim, self.arena.ydim), interpolation=cv2.INTER_AREA)
            self.pic = resized
            
            # Count this frame
            self.arena.frames += 1
            self.total_frames_generated += 1
            
            # Send frame to save queue if enabled
            if self.save_queue is not None and self.total_frames_generated <= self.arena.frames:
                frame_metadata = {
                    'timestamp': time.time(),
                    'frame_number': self.total_frames_generated - 1,  # 0-based frame numbers
                    'frame_data': self.pic.copy()
                }
                self.save_queue.put(frame_metadata)
                self.saved_frames += 1
                if self.saved_frames % 60 == 0 and self.debug:
                    print(f"[Frames] Generated={self.total_frames_generated}, Saved={self.saved_frames}")
        else:
            # If it's not time for a new frame yet, show the previous frame
            self.pic = self.arena.oldframe
        
        # Remember this frame for the next iteration
        self.arena.oldframe = self.pic
        
        return self.pic
    

class ShowVerticalEdge():
    
    def __init__(self,arena):
         
    
        self.arena = arena # object of the class Stimulation_Pipeline() 
        # This step is necessary in order to use functions and variables, such as time, from the stimulation pipeline class 
        # and to generate a separate initialisation function and a run function independently for each stimulus.
        xdim=self.arena.xdim
        ydim=self.arena.ydim
        self.pic = np.zeros([ydim,xdim,3],dtype = "uint8")
        
        fpss = np.array([])
        fps = 0
        
        timer = cv2.getTickCount()
        

        
    def run(self,start,speed,color1,color2):
                        
        
        self.arena.dt = time.time()-self.arena.time_start 
        pixel_start = start/self.arena.resolution[1]
        pixel_shifted = (start+self.arena.dt*speed)/self.arena.resolution[1]
        pic = np.ones([self.arena.ydim,self.arena.xdim],dtype = "uint8")*color1
        pic[:,int(pixel_start):int(pixel_shifted)]=color2
        self.pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)    
        
        return self.pic
    
 ######################################### UNTESTED ###########################################

