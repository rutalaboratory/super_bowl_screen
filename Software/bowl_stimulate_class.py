import numpy as np
import matplotlib.pyplot as plt
from bowl import *
import time
import sys
import functools



class Stimulation_Pipeline():
    
    def __init__(self,img_size=(360, 720,3), fov_azi=(0,180), fov_ele=(15,140),img_offsetx=3840+3240,img_offsety=2400,name = "Arena", projector_width_pixels=1280):
        #initialize Projection Objects
        azi_pix = int(img_size[1]/360*fov_azi[1])
        ele_pix = int(img_size[0]/180*fov_ele[1])
        
        print("Azi Pix", azi_pix)
        print("ele Pix", ele_pix)

        self.azi_pix = azi_pix
        self.ele_pix = ele_pix
        
        self.xdim = img_size[1]
        self.ydim = img_size[0]
        self.resolution = np.array([1/(self.ele_pix/fov_ele[1]),1/(self.azi_pix/fov_azi[1])])
        self.image_size = img_size
        self.dest = np.zeros(img_size,dtype = "uint8")
        self.Stimulus = Stimulus(self.dest.shape)
        self.Projector_1 = Projector(proj_x=projector_width_pixels, proj_y=int(projector_width_pixels/2))
        self.Projector_1.initialize_projection_matrix((ele_pix,azi_pix),fov_azi,fov_ele)
        self.dt = 0
        self.time_start =0
        self.frames=0
        self.oldframe =0
        

        #initialize Window output

        self.WINDOW_NAME = name
        self.width_first  = img_offsetx
        self.height_first = img_offsety
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.WINDOW_NAME, self.width_first, self.height_first)
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("initialize Stimulation Pipeline < ",self.WINDOW_NAME," >: xdim=,",self.xdim,"ydim=",self.ydim,
              "at position x=", self.width_first,"y=", self.height_first)
        # generate stimulus texture 

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
        key = cv2.waitKey(30)

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
        print (np.mean(fpss))
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
        print (np.mean(fpss))
        
        
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
    
class MovingDotAzimuthal():
    
    def __init__(self, arena, elevation=90, elevation_width=5, azi_limits=(30,150), speed=20):
        self.arena = arena
        self.elevation = elevation
        self.elevation_width = elevation_width
        self.azi_min, self.azi_max = azi_limits
        self.speed = speed  # degrees/sec
        self.direction = 1  # 1 for right, -1 for left

        # calculate pixel radius based on elevation width
        self.pixel_radius = int((elevation_width / 2) / self.arena.resolution[0])

        # fixed pixel y position based on elevation
        self.y = int((1 - (elevation - self.arena.Projector_1.fov_ele[0]) /
                    (self.arena.Projector_1.fov_ele[1] - self.arena.Projector_1.fov_ele[0])) * self.arena.ydim)

        # start at min azimuth
        self.azi = self.azi_min
        self.last_update_time = time.time()
        self.pic = np.full((self.arena.ydim, self.arena.xdim, 3), 255, dtype=np.uint8)  # white background


    def run(self):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update azimuth based on speed and direction
        self.azi += self.speed * dt * self.direction

        # Reflect direction if hitting bounds
        if self.azi >= self.azi_max:
            self.azi = self.azi_max
            self.direction = -1
        elif self.azi <= self.azi_min:
            self.azi = self.azi_min
            self.direction = 1

        # Calculate x position in pixels
        x = int((self.azi - self.arena.Projector_1.fov_azi[0]) /
                (self.arena.Projector_1.fov_azi[1] - self.arena.Projector_1.fov_azi[0]) * self.arena.xdim)

        # Reset image
        self.pic[:, :, :] = 255  # white background
        cv2.circle(self.pic, (x, self.y), self.pixel_radius, (0, 0, 0), -1)

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
    
    def __init__(self,arena,pixelsize,framerate=30):
         
    
        self.arena = arena # object of the class Stimulation_Pipeline() 
        # This step is necessary in order to use functions and variables, such as time, from the stimulation pipeline class 
        # and to generate a separate initialisation function and a run function independently for each stimulus.
        xdim=self.arena.xdim
        ydim=self.arena.ydim
        self.pic = np.zeros([ydim,xdim,3],dtype = "uint8")
        self.y_noise = self.arena.ele_pix*self.arena.resolution[0]/pixelsize
        self.x_noise = self.arena.azi_pix*self.arena.resolution[1]/pixelsize
        self.arena.frames=0
        print("video framerate =",framerate)
        self.framerate = framerate
        fpss = np.array([])
        fps = 0
        print("y noise pixel = ",self.y_noise, " x noise pixel = ",self.x_noise)
        np.random.seed(0)
        timer = cv2.getTickCount()
        

        
    def run(self):
                        
        theoretical_elapsed_time = self.arena.frames*(1/self.framerate)
        self.arena.dt = time.time()-self.arena.time_start 
        #print("dt = ",self.arena.dt,"  frames = ",self.arena.frames)
        
        if self.arena.dt>=theoretical_elapsed_time:

            Input_im = (np.random.randint(0,2,(int(self.y_noise ),int(self.x_noise),1))*255).astype("uint8")
            image = cv2.cvtColor(Input_im, cv2.COLOR_GRAY2RGB)    
            resized = cv2.resize(image,dsize=(self.arena.azi_pix, self.arena.ele_pix), interpolation = cv2.INTER_AREA)
            self.pic[0:280,180:540,:]= resized
 
            self.arena.frames += 1
        else:
            self.pic = self.arena.oldframe
            
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

