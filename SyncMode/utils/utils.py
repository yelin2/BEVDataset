
import glob
import os
import sys
import time
import math

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
from utils import carla_vehicle_BEV as cva

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


def arg(argparser):
    # =====================================
    #         carla client settings
    # =====================================
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    
    argparser.add_argument(
        '-nov', '--number-of-vehicles',
        metavar='N',
        default=50,
        type=int,
        help='number of vehicles (default: 10)')
    
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    
    
    # =====================================
    #       carla fixed seed settings
    # =====================================
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        default=1,
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    
    argparser.add_argument(
        '--show',
        metavar='show',
        default=False,
        help='show camera image with pygame window ')
    
    argparser.add_argument(
        '-d','--debug',
        metavar='debug',
        default=False,
        help='for debuging')
    
    
    # =====================================
    #            camera settings
    # =====================================
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')

    argparser.add_argument(
        '--w',
        metavar='W',
        default=964,
        type=int,
        help='cam setting(w)')
    
    argparser.add_argument(
        '--h',
        metavar='H',
        default=964,
        type=int,
        help='cam setting(h)')
    
    argparser.add_argument(
        '--x_meter',
        metavar='X',
        default=85,
        type=int,
        help='cam setting(x_meter)')
    
    argparser.add_argument(
        '--fov',
        metavar='F',
        default=70.4,
        type=int,
        help='cam setting(fov)')
    
    
    # =====================================
    #              save settings
    # =====================================
    argparser.add_argument(
        '--data_root',
        default='/home/rml/ws/TestData',
        type=str,
        help='root data path, change if local machine changed')

    argparser.add_argument(
        '--scene_num',
        default='/drive_001',
        type=str,
        help='driving scene number, change if driving scene changed')

    argparser.add_argument(
        '--is_static',
        default=False,
        type=bool,
        help='True if static object save')

    argparser.add_argument(
        '--include_removed',
        default=False,
        type=bool,
        help='True if save occluded bbox')

    argparser.set_defaults(sync=True)
    
    return argparser

def retrieve_data(sensor_queue, frame, timeout=1):
    while True:
        try:
            data = sensor_queue.get(True,timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data
    

def metersetting(w, h, x_meter,  fov):
    '''
        w, h, x_meter, fov -> cam height & meter per pixel calculate
    '''

    x_rad = np.radians(fov/2)
    x_meter = x_meter/2
    cam_height = x_meter/np.tan(x_rad)

    x_mpp = x_meter/w
    y_meter = x_mpp * h
    return cam_height, x_mpp, y_meter


def draw_arrow(world, waypoints = None, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    if waypoints is None:
        location = carla.Location(0,0,0)
        rotation = carla.Rotation(0,0,0)
        begin = location + carla.Location(z=1)
        angle = math.radians(rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end,thickness=0.2, arrow_size=0.2, life_time=0)

        location = carla.Location(0,10,0)
        rotation = carla.Rotation(0,90,0)
        begin = location + carla.Location(z=1)
        angle = math.radians(rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, thickness=0.2,arrow_size=0.2,color=carla.Color(0,0,255), life_time=0)

        location = carla.Location(10,0,0)
        rotation = carla.Rotation(0,-90,0)
        begin = location + carla.Location(z=1)
        angle = math.radians(rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, thickness=0.2,arrow_size=0.2,color=carla.Color(0,255,0), life_time=0)

    else: 
        for wpt in waypoints:
            begin = wpt.location + carla.Location(z=z)
            angle = math.radians(wpt.rotation.yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=0)

def draw_points(world, spawnpoints, color = [255,255,0], point_size = 0.1, time = 10, is_list = 1):
    """
    Draw a list of spawnpoints at a certain height given in z.

        :param world: carla.world object
        :param spawnpoints: list or iterable container with the spawnpoints to draw
        :param z: height in meters
    """
    if is_list:
        for i, spt in enumerate(spawnpoints):
            location = spawnpoints[i].location
            print(location)

            world.debug.draw_point(location, size=point_size, color=carla.Color(color[0], color[1], color[2]), life_time=time)
  
    else:
        location = spawnpoints.location
        print(location)

        world.debug.draw_point(location, size=point_size, color=carla.Color(color[0], color[1], color[2]), life_time=time)
    

def show_od_image(vehicles_raw, snap, depth_img, rgb_img, t_depth):
    
    # Attach additional information to the snapshot
    vehicles = cva.snap_processing(vehicles_raw, snap)

    # Save depth image, RGB image, and Bounding Boxes data
    depth_meter = cva.extract_depth(depth_img)

    vehicle_filtered, vehicle_removed =  cva.auto_annotate(vehicles, t_depth.sensor, depth_meter, cls='vehicle')
    vehicle_box_rgb = cva.save_output(rgb_img, vehicle_filtered['bbox'], vehicle_filtered['class'], \
        vehicle_removed['bbox'], vehicle_removed['class'], save_patched=True, out_format='json')

    # pygame display
    if t_depth.display_man.render_enabled():
        vehicle_box_rgb = cv2.resize(vehicle_box_rgb, (350, 350)) 
        t_depth.surface = pygame.surfarray.make_surface(vehicle_box_rgb.swapaxes(0, 1))
                
