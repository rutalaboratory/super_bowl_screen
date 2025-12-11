import numpy as np
import matplotlib.pyplot as plt
from bowl import *
import time
import sys
import functools

import fictrac


class Stimulation_Pipeline():
    """
    The Stimulation_Pipeline class manages the display of visual stimuli in a on the spherical screen.
    It handles the conversion from regular images to the distorted projections needed for the bowl shape,
    and manages the display window positioning and timing.
    """
    
    def __init__(self,img_size=(360, 720,3),
                  fov_azi=(0,180), 
                  fov_ele=(15,140),
                  monitor_resolution=(1920, 1080),
                  projector_resolution=(1280, 720),
                  name = "Arena",
                  projector_width_pixels=1280, 
                  arduino=None,
                  dark_screen_duration=0.1,
                  debug=False):
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
        
        # set the baseline period (dark screen) before the stimulus presentation
        self.dark_screen_duration = dark_screen_duration

        # initialize the arduino object
        self.arduino = arduino
        self.arduino_handshake_flag = False

        self.image_width = img_size[1]  # Width of the image in pixels
        self.image_height = img_size[0]  # Height of the image in pixels

        self.azimuthal_maximum = fov_azi[1]  # Maximum azimuthal angle in degrees
        self.azimuthal_minimum = fov_azi[0]  # Minimum azimuthal angle in degrees

        self.elevation_maximum = fov_ele[1]  # Maximum elevation angle in degrees
        self.elevation_minimum = fov_ele[0]  # Minimum elevation angle in degrees

        # Calculate how many pixels we need for the given field of view
        # For example, if we want to show 180 degrees in 720 pixels, each degree needs 4 pixels
        # The sphere wraps around 360 degrees horizontally and 180 degrees vertically
        azi_pix = int((self.image_width/360) * self.azimuthal_maximum)  # Pixels per degree in azimuth
        ele_pix = int((self.image_height/180) * self.elevation_maximum)  # Pixels per degree in elevation

        if self.debug: 
            print(azi_pix, ele_pix)

        # Store pixel dimensions for later use
        self.azi_pix = azi_pix  # Width in pixels
        self.ele_pix = ele_pix  # Height in pixels
        
        # Store image dimensions and calculate resolution (degrees per pixel)
        self.xdim = self.image_width  # Image width
        self.ydim = self.image_height  # Image height

        # Calculate resolution in degrees per pixel
        self.resolution = np.array([1/(self.ele_pix/self.elevation_maximum),
                                    1/(self.azi_pix/self.azimuthal_maximum)])  # Degrees per pixel
        print("Pixel Resolution (degrees per pixel) (x, y): ", self.resolution)

        self.image_size = img_size  # Full image size including color channels
        
        # Create a blank image buffer for stimulus generation
        self.dest = np.zeros(img_size,dtype = "uint8")
        
        # Initialize the components that handle stimulus generation and projection
        self.Stimulus = Stimulus(img_size=img_size, 
                                 fov_azi=fov_azi, 
                                 fov_ele=fov_ele)  # Handles stimulus creation and rotation

        # Set up projector parameters (this handles the bowl-shape distortion)
        # proj_x is the width of the projected image that fits the bowl
        # proj_y is half the width to maintain aspect ratio
        # Here we assume a 2:1 aspect ratio for the bowl projection

        self.projector_width_pixels = projector_width_pixels

        self.Projector_1 = Projector(res_x=projector_resolution[0],
                                     res_y=projector_resolution[1],
                                     proj_x=self.projector_width_pixels,
                                     proj_y=int(self.projector_width_pixels/2),
                                     fov_azi=fov_azi,
                                     fov_ele=fov_ele)

        self.Projector_1.initialize_projection_matrix((ele_pix,azi_pix),
                                                      fov_azi,
                                                      fov_ele)
        
        self.warmup_jit()   # Warm up JIT-compiled functions
        print("JIT warmup complete.")

        # Initialize timing variables for stimulus presentation
        self.dt = 0          # Time elapsed since start
        self.time_start = 0  # Start time of stimulus
        self.frames = 0      # Number of frames shown
        self.oldframe = 0    # Previous frame (used for timing)
        
        self.is_new_frame = True  # Flag to indicate if a new frame was generated

        # Initialize data structures for capturing data from HorizontalMovingDot
        self.total_elapsed_time = None  # Total elapsed time since start
        self.yaw = None  # Yaw angle (to be updated by stimulus)
        self.azimuthal_position = None  # Azimuthal position of the dot
        self.elevational_position = None  # Elevational position of the dot
        self.fictrac_data = None  # Placeholder for FicTrac data

        # Set up the display window for the stimulus
        # This creates a window on the second monitor at the specified position

        self.WINDOW_NAME = name          # Name of the window
        self.width_first = monitor_resolution[0] + projector_resolution[0]   # X position of window (usually on second monitor)
        self.height_first = monitor_resolution[1] + projector_resolution[1]   # Y position of window
        
        # Create and position the window
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.moveWindow(self.WINDOW_NAME, self.width_first, self.height_first)  # Move it to the right position
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Make it fullscreen
        
        # Print initialization info
        print("initialize Stimulation Pipeline < ",self.WINDOW_NAME," >: xdim=,",self.xdim,"ydim=",self.ydim,
              "at position x=", self.width_first,"y=", self.height_first)

    def warmup_jit(self):
        # Dummy image with the right shape
        dummy_img = np.zeros((int(self.projector_width_pixels/2), int(self.projector_width_pixels), 3), dtype=np.uint8)
        dummy_dest = np.zeros_like(dummy_img)
        
        # Warm up Stimulus rotation
        _ = self.Stimulus.rot_equi_img(dummy_img, dummy_dest, 0.0, 0.0, 0.0)

        # Warm up projection
        _ = self.Projector_1.project_image(dummy_img)

        # Warm up mask insertion
        _ = self.Projector_1.mask_image(dummy_img)

        # Warm up select_fov / write_fov
        _ = select_fov(dummy_img)

        # Call with typical arguments to trigger JIT compilation
        _ = rotation_matrix(0.0, 0.0, 0.0)

        # Warm up get_dot_coordinates with typical yaw value
        # dummy_dot = HorizontalMovingDot(self, dot_initial_position=0)
        # _ = dummy_dot.get_dot_coordinates(yaw=0.0)



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
        # key = cv2.waitKey(int(duration*1000))

        start = time.perf_counter()

        while (time.perf_counter() - start) < duration:
            key = cv2.waitKey(1)  # check every 1 ms
            if key & 0xFF == 27:  # ESC

                cv2.destroyAllWindows()
                return False  # aborted

        return True  # finished normally


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
            cv2.imshow("croped",np.array(croped))
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
     
        
    def generate(self, function, duration=0, rot_offset=(0,0,0), save_queue=None, *args, **kwargs):
        """
        Display frames produced by `function` for `duration` seconds, projecting them to the bowl.
        While running, save for each displayed frame:
        - the raw frame returned by `function` (Input_im),
        - the display timestamp (taken immediately after cv2.waitKey),
        - a sequential frame number (0-based).

        Pass your queue like:
            Arena.generate(noise.run, duration=..., rot_offset=(0,0,0), save_queue=noise_queue)
        """
        # Pre-run trigger/dark screens just like before
        #self.show_trigger()
        #self.show_dark_screen(0.1)
        fpss = np.array([])
        fps = 0
        timer = cv2.getTickCount()
       
        if not self.show_dark_screen(self.dark_screen_duration):
            print("[Image Generator] Aborted before starting (ESC pressed during dark screen).")
            return
        
        # self.show_dark_screen(self.dark_screen_duration)
        #self.show_trigger()
        self.time_start = time.perf_counter_ns() / 1e9  # Convert to seconds

        # Local counter for saved frame numbers
        saved_frame_idx = 0
                
        while (time.perf_counter_ns() / 1e9) < self.time_start + (duration - self.dark_screen_duration):
            # Get the next frame from the generator/callback
            Input_im = function(*args, **kwargs)

            if hasattr(function.__self__, "yaw") and hasattr(function.__self__, "get_dot_coordinates"):
                az, el = function.__self__.get_dot_coordinates(function.__self__.yaw, rot_offset)
                # print(f"Dot position -> Azimuth: {az:.2f}°, Elevation: {el:.2f}°")


            # print("Timestamp after Image Received: ", time.perf_counter_ns())

            # Ensure 3 channels for projection
            if len(Input_im.shape) < 3:
                resized = cv2.cvtColor(Input_im, cv2.COLOR_GRAY2RGB)
            else:
                resized = Input_im

            # Optional static rotation offset
            if rot_offset == (0, 0, 0):
                rotated = resized
            else:
                rotated = self.Stimulus.rot_equi_img(resized, self.dest, rot_offset[0], rot_offset[1], rot_offset[2])

            # print("Timestamp after Image Rotated: ", time.perf_counter_ns())

            # Project to bowl and show
            cropped = select_fov(rotated)

            # print("Timestamp after Image Cropped: ", time.perf_counter_ns())

            masked = self.Projector_1.project_image(cropped)
            
            # print("Timestamp after Image Projected: ", time.perf_counter_ns())

            output = self.Projector_1.mask_image(masked)

            # print("Timestamp after Image Masked: ", time.perf_counter_ns())

            # --- Arduino Handshake: once per new frame ---
            if self.arduino is not None and self.is_new_frame:
                self.arduino.handshake()
            
            timestamp =  time.perf_counter_ns()  # Timestamp in nanoseconds

            cv2.imshow(self.WINDOW_NAME, output)

            # Update FPS stats
            tick = cv2.getTickCount() - timer
            fps = cv2.getTickFrequency() / (tick)
            timer = cv2.getTickCount()
            fpss = np.append(fpss, fps)

            # Let OpenCV update the window; time *after* this approximates display time
            key = cv2.waitKey(1)

            elapsed_time = (time.perf_counter_ns() / 1e9) - self.time_start
            # print(f"[Image Generator] Elapsed time: {elapsed_time:.2f}s / {duration:.2f}s", end='\r', flush=True)

            if self.debug:
                print('[Image Generator] Frame', saved_frame_idx, 'at', timestamp, 'ns')
            
            # ---- Save data here  ----
            if save_queue is not None:
                save_queue.put({
                    'original_image': Input_im.copy() if hasattr(Input_im, 'copy') else Input_im,
                    'bowl_image': output.copy() if hasattr(output, 'copy') else output,
                    'timestamp': timestamp,
                    'frame_number': saved_frame_idx,
                    'is_new_frame': int(self.is_new_frame),  # 1 if new, 0 if reused
                    'azimuthal_position': getattr(self, 'azimuthal_position', None),
                    'elevation_position': getattr(self, 'elevation_position', None),
                    'yaw': getattr(self, 'yaw', None),
                    'total_elapsed_time': getattr(self, 'total_elapsed_time', None),
                    'fictrac_data': getattr(self, 'fictrac_data', None)
                })
                saved_frame_idx += 1

            if key == 27:  # ESC to exit early
                cv2.destroyAllWindows()
                break

        # self.show_trigger()
        # self.show_dark_screen(0.1)
        print(f"[Image Generator] Average Loop FPS: {np.mean(fpss):.1f}, Total frames displayed: {saved_frame_idx}, Total New frames generated: {self.frames}")
        
        
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
    
    def generate_dot(self, background_color, dot_color, dot_center, dot_radius, debug=False):
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

        if debug:
            cv2.imshow("dot pos", img)
            key = cv2.waitKey(1)

        return img

    def horizontal_projection(self):

        img = np.ones((self.ydim, self.xdim), dtype=np.uint8) * 255
        print("bugbug", self.xdim,self.ydim)
        cv2.circle(img, (int(self.xdim/2), 280), 40, 0, -1)

        cv2.imshow("dot pos", img)
        key = cv2.waitKey(1)

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
    
    def __init__(self, arena, pixelsize, framerate=30, debug=False):
        """
        Set up the noise stimulus with the given parameters.
        
        Args:
            arena: The projection environment where the stimulus will be shown
            pixelsize: Size of each noise pixel (larger numbers = bigger chunks of noise)
            framerate: How many times per second the noise pattern updates (default: 30Hz)
            debug: Enable debug output
        """
        # Store reference to the arena for accessing screen dimensions and timing
        self.debug = debug
        self.arena = arena
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

    # In class ShowNoise 
    def run(self):
        """
        Generate and return the next frame of noise.
        """
        # Calculate when the next frame should be shown based on the desired framerate
        theoretical_elapsed_time = self.arena.frames * (1 / self.framerate)
        self.arena.dt = (time.perf_counter_ns() / 1e9) - self.arena.time_start

        # Print timing info every 60 frames
        if self.arena.frames % 60 == 0 and self.debug:
            timing_diff = self.arena.dt - theoretical_elapsed_time
            print(f"[Timing] Frame {self.arena.frames}: Real time={self.arena.dt:.3f}s, "
                f"Theoretical={theoretical_elapsed_time:.3f}s, "
                f"Diff={timing_diff*1000:.1f}ms")

        self.arena.is_new_frame = False

        # If it's time for a new frame...
        if self.arena.dt >= theoretical_elapsed_time:
            # Create a new random noise pattern
            Input_im = (np.random.randint(0, 2, (int(self.y_noise), int(self.x_noise), 1)) * 255).astype("uint8")

            # Convert to RGB and scale to arena size
            image = cv2.cvtColor(Input_im, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(image,dsize=(self.arena.azi_pix, self.arena.ele_pix), interpolation = cv2.INTER_AREA)

            self.pic[0:280,180:540,:]= resized

            # Count this frame
            self.arena.frames += 1
            self.arena.is_new_frame = True
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

import numpy as np
import cv2
import time

class ShowPattern():
    """
    This class creates a visual stimulus that displays a deterministic pattern 
    of colored stripes on the projection screen. The stripes are repeatable and 
    predictable, unlike random noise.
    """
    
    def __init__(self, arena, pixelsize, framerate=30, save_queue=None, debug=False, 
                 orientation="vertical", moving=False):
        """
        Set up the stripe stimulus with the given parameters.
        
        Args:
            arena: The projection environment where the stimulus will be shown
            pixelsize: Size of each stripe element (larger = thicker stripes)
            framerate: How many times per second the pattern updates (default: 30Hz)
            save_queue: Optional multiprocessing queue to save generated patterns
            orientation: "vertical" or "horizontal" stripes
            moving: If True, stripes will shift across frames
        """
        self.debug = debug
        self.arena = arena
        self.save_queue = save_queue
        self.orientation = orientation
        self.moving = moving
        
        xdim = self.arena.xdim
        ydim = self.arena.ydim
        
        self.pic = np.zeros([ydim, xdim, 3], dtype="uint8")
        
        # How many "cells" based on pixelsize
        self.y_cells = int(self.arena.ele_pix * self.arena.resolution[0] / pixelsize)  
        self.x_cells = int(self.arena.azi_pix * self.arena.resolution[1] / pixelsize)  
        
        # Initialize frame counting and timing
        self.arena.frames = 0
        self.framerate = framerate
        print("video framerate =", framerate)
        print("y cells = ", self.y_cells, " x cells = ", self.x_cells)
        
        print("DEBUG", xdim, ydim, self.x_cells, self.y_cells)

        # Define a repeating sequence of colors (BGR for OpenCV)
        self.colors = [
            (255, 0, 0),     # Blue
            (0, 255, 0),     # Green
            (0, 0, 255),     # Red
            (0, 255, 255),   # Yellow
            (255, 0, 255),   # Magenta
            (255, 255, 0)    # Cyan
        ]

    def _generate_stripes(self, frame_idx):
        """
        Create colored stripes (vertical or horizontal).
        If moving=True, shift the stripe pattern each frame.
        """
        pattern = np.zeros((self.y_cells, self.x_cells, 3), dtype="uint8")
        
        if self.orientation == "vertical":
            for j in range(self.x_cells):
                color_idx = (j + (frame_idx if self.moving else 0)) % len(self.colors)
                pattern[:, j, :] = self.colors[color_idx]
        
        elif self.orientation == "horizontal":
            for i in range(self.y_cells):
                color_idx = (i + (frame_idx if self.moving else 0)) % len(self.colors)
                pattern[i, :, :] = self.colors[color_idx]
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
        
        return pattern

    def run(self):
        """
        Generate and return the next frame of deterministic colored stripes.
        """
        theoretical_elapsed_time = self.arena.frames * (1 / self.framerate)
        self.arena.dt = time.time() - self.arena.time_start

        if self.arena.frames % 60 == 0 and self.debug:
            timing_diff = self.arena.dt - theoretical_elapsed_time
            print(f"[Timing] Frame {self.arena.frames}: Real time={self.arena.dt:.3f}s, "
                f"Theoretical={theoretical_elapsed_time:.3f}s, "
                f"Diff={timing_diff*1000:.1f}ms")

        if self.arena.dt >= theoretical_elapsed_time:
            # Generate stripes
            pattern = self._generate_stripes(self.arena.frames)

            # Resize to arena
            resized = cv2.resize(pattern, dsize=(self.arena.azi_pix, self.arena.ele_pix), 
                                 interpolation=cv2.INTER_NEAREST)

            # Place in display area
            self.pic[0:280, 180:540, :] = resized

            self.arena.frames += 1
        else:
            self.pic = self.arena.oldframe

        cv2.imshow("Pattern Debug", self.pic)
        key = cv2.waitKey(1)    
        self.arena.oldframe = self.pic

        return self.pic

# Functions for horizontal moving dot stimulus

import jax
import jax.numpy as jnp

@jax.jit
def rotation_matrix(roll=0, pitch=0, yaw=0):
    roll, pitch, yaw = jnp.deg2rad(jnp.array([roll, pitch, yaw]))
    R_x = jnp.array([[1, 0, 0],
                    [0, jnp.cos(roll), -jnp.sin(roll)],
                    [0, jnp.sin(roll),  jnp.cos(roll)]])
    R_y = jnp.array([[ jnp.cos(pitch), 0, jnp.sin(pitch)],
                    [0, 1, 0],
                    [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    R_z = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],
                    [jnp.sin(yaw),  jnp.cos(yaw), 0],
                    [0, 0, 1]])
    return R_x @ R_y @ R_z

class HorizontalMovingDot():
    def __init__(self,
                 arena,
                 dot_initial_position=0,
                 dot_cooldown=60,
                 dot_size=40, 
                 dot_speed=10, 
                 dot_direction=1,
                 dot_limits=(-140, 140),
                 fictrac_params={"host": "127.0.0.1", "port": 3000},
                 debug=False):
        
        self.arena = arena  # object of the class Stimulation
        
        # fictrac parameters
        self.fictrac_params = fictrac_params
        print("fictrac params:", self.fictrac_params)
        self.fictrac_port = fictrac_params["port"]
        self.fictrac_host = fictrac_params["host"]

        # stimulus parameters
        self.dot_initial_position = dot_initial_position  # initial yaw position in degrees
        self.dot_cooldown = dot_cooldown  # time to wait before starting movement
        self.dot_size = dot_size  # diameter in pixels
        self.dot_speed = dot_speed  # deg/sec
        self.dot_direction = dot_direction  # 1: right to left, -1: left to right
        self.dot_limits = dot_limits  # min and max yaw positions in degrees
        self.debug = debug

        # initialize stimulus image
        self.pic = np.ones([self.arena.ydim, self.arena.xdim], dtype="uint8") * 255
        self.pic[0:int(self.dot_size), :] = 0  # black line at the top (north pole)
        self.pic = np.stack([self.pic] * 3, axis=-1)

        self.Stim_init = self.arena.Stimulus.rot_equi_img(
            self.pic, self.arena.dest, roll=0, pitch=-90, yaw=0)

        # local timer and yaw tracking
        self.yaw = self.dot_initial_position   # current yaw
        print("yaw initialized to:", self.yaw)
        # az, el = self.get_dot_coordinates(self.yaw, rot_offset=(0, 90, 0))
        
        # print(f"Initial dot position -> Azimuth: {az:.2f}°, Elevation: {el:.2f}°")

         # initialize the fictrac client 
        self.fictrac_client = fictrac.FicTracClient(host=self.fictrac_host, port=self.fictrac_port)
        self.fictrac_client.connect()
        print("FicTrac client initialized.")

    def run(self):
   
        # On the first call, initialize last_time from arena.time_start
        if not hasattr(self, '_initialized_last_time'):
            self.last_time = self.arena.time_start
            self._initialized_last_time = True

        self.total_elapsed_time = time.perf_counter_ns() / 1e9 - self.arena.time_start
        self.arena.dt = time.perf_counter_ns() / 1e9 - self.last_time

        # print("dt obtained by run:", self.arena.dt)

        # read data from fictrac
        fictrac_data = self.fictrac_client.read_frame()
        
        self.current_heading = fictrac_data['heading']  # in degrees
        
        if self.debug:
            print("FicTrac heading:", self.current_heading)

        # fictrac_data = None
        if self.total_elapsed_time > self.dot_cooldown:
            self.protocol("closed_loop")
            # print(f"Waiting for cooldown: {self.dot_cooldown - self.total_elapsed_time:.2f}s remaining",  end='\r')

            print("yaw =", self.yaw)

        # update stimulus image
        self.pic = self.arena.Stimulus.rot_equi_img(
            self.Stim_init,
            self.arena.dest,
            roll=0,
            pitch=0,
            yaw=self.yaw
        )
        
        self.arena.azimuthal_position, self.arena.elevation_position = self.get_dot_coordinates(self.yaw, rot_offset=(0, 90, 0))
        self.arena.total_elapsed_time = self.total_elapsed_time  
        self.arena.yaw = self.yaw 
        self.arena.fictrac_data = fictrac_data

        self.last_time = time.perf_counter_ns() / 1e9  # update local timer

        return self.pic
    
    def get_dot_coordinates(self, yaw=0, rot_offset=(0,0,0)):
        return get_dot_coordinates_jit(yaw, rot_offset)
    
    def protocol(self, type):
        if type == "open_loop":
            # update yaw incrementally
            # print("Time difference:", self.arena.dt)
            self.yaw += self.dot_direction * self.dot_speed * self.arena.dt
            # print("yaw obtained by run:", self.yaw)
            # flip direction at edges
            if self.yaw > self.dot_limits[1]:
                self.yaw = self.dot_limits[1]
                self.dot_direction = -1
            elif self.yaw < self.dot_limits[0]:
                self.yaw = self.dot_limits[0]
                self.dot_direction = 1
        else:
            self.yaw = self.dot_initial_position
        
        if type == "closed_loop":
            self.current_heading = self.current_heading * (180 / jnp.pi)  # convert from radians to degrees
            self.yaw = -self.current_heading
            # in closed-loop, yaw is controlled by fictrac heading

@jax.jit
def get_dot_coordinates_jit(yaw=0, rot_offset=(0,0,0)):
    """
    Convert a dot's yaw position into final (azimuth, elevation) in degrees,
    accounting for:
    - software pitch = -90° (for round dot)
    - physical setup pitch = -90° (north pole is at equator)
    - Yaw is flipped 180° on the bowl
    - global rot_offset from Arena.generate
    """
    # Start at north pole
    p = jnp.array([0.0, 0.0, 1.0])

    # Software pitch to make dot round
    p = rotation_matrix(pitch=-90) @ p

    # Apply yaw rotation 
    # The yaw rotation is flipped 180° on the bowl
    p = rotation_matrix(yaw=yaw) @ p
    p = rotation_matrix(yaw=180) @ p
    
    # Apply physical setup pitch offset (-90°)
    p = rotation_matrix(pitch=-90) @ p

    # Apply global offset from Arena.generate
    p = rotation_matrix(*rot_offset) @ p

    # Normalize to unit vector
    p = p / jnp.linalg.norm(p)

    # Convert to spherical coordinates
    azimuth = jnp.rad2deg(jnp.arctan2(p[1], p[0]))
    elevation = jnp.rad2deg(jnp.arcsin(p[2]))

    return azimuth, elevation

