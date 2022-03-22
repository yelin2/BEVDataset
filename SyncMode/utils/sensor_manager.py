import glob
import os
import sys

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.7-%s.egg' % (
        sys.version_info.major,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


import queue
import numpy as np
import cv2


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')



class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options=None, display_pos=None, show = False):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.q = queue.Queue()
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options

        self.display_man.add_sensor(self)

        self.show = show
    

    def init_sensor(self, sensor_type, transform, attached, sensor_options=None):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'DEPTHCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_depth_image)

            return camera
        
        elif sensor_type == 'SEMANTICCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_semantic_image)

            return camera
        
        elif sensor_type == 'GNSS':
            gps_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
            gps = self.world.spawn_actor(gps_bp, transform, attach_to=attached, attachment_type=carla.AttachmentType.Rigid)
            gps.listen(self.save_gps)

            return gps

        elif sensor_type == 'IMU':
            imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
            imu = self.world.spawn_actor(imu_bp, transform, attach_to=attached, attachment_type=carla.AttachmentType.Rigid)
            imu.listen(self.save_data)
            return imu

        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image, show = False):
        self.q.put(image)

        if self.show:

            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = cv2.resize(array, (350, 350)) 

            if self.display_man.render_enabled():
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def save_depth_image(self, image):
        self.q.put(image)

    def save_data(self, data):
        self.q.put(data)

    def save_gps(self, data):
        self.q.put(data)

    def save_semantic_image(self, image):
        self.q.put(image)


        if self.show:

            image.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = cv2.resize(array, (350, 350)) 


            if self.display_man.render_enabled():
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()