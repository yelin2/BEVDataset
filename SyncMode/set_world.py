### Example program to save several sensor data including bounding box
### Sensors: RGB Camera (+BoundingBox), De[th Camera, Segmentation Camera, Lidar Camera
### By Mukhlas Adib
### 2020
### Last tested on CARLA 0.9.10.1

### CARLA Simulator is licensed under the terms of the MIT license
### For a copy, see <https://opensource.org/licenses/MIT>
### For more information about CARLA Simulator, visit https://carla.org/


import glob
import os
import sys
from tracemalloc import Snapshot

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.7-%s.egg' % (
        sys.version_info.major,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random
import queue

from utils.utils import *
from utils.display_manager import DisplayManager
from utils.sensor_manager import SensorManager
from utils.save_utils import save_seg, load_trajectory


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')



def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def run_simulation(args, client):
    # -----------------------------
    #         pygame setting
    # -----------------------------
    pygame.init()
    screen = pygame.display.set_mode((1700,700)) # w, h
    pygame.display.set_caption('vehicle segmentation')
    done = True
    clock = pygame.time.Clock()

    vehicles_list = []

    try:
        # ===============================================================================
        #                               Set World - Fixed Seed
        # ===============================================================================

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)
            random.seed(args.seed)

        # world = client.load_world('Town02')      #! need to set maps
        world = client.get_world()
        original_settings = world.get_settings()

        # -----------------------------
        #       Set layered map
        # -----------------------------
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        world.unload_map_layer(carla.MapLayer.Props)
        world.unload_map_layer(carla.MapLayer.StreetLights)
        world.unload_map_layer(carla.MapLayer.Decals)
        world.unload_map_layer(carla.MapLayer.Foliage)



        # ===============================================================================
        #                                   Begin the loop
        # ===============================================================================
        cnt = 0
        time_sim = 0
        timestamps = []
        call_exit = False

        tick_sensor = 0

        while True:
            # Extract the available data
            nowFrame = world.tick()
            

            # Check whether it's time to capture data
            if time_sim >= tick_sensor:
                
                clock.tick(60)
                
                time_sim = 0

            time_sim = time_sim + settings.fixed_delta_seconds

    finally:


        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        world.apply_settings(original_settings)

def main():

    # -----------------------------
    #           argument
    # -----------------------------
    argparser = argparse.ArgumentParser(
            description=__doc__)
    argparses = arg(argparser)
    args = argparses.parse_args()

    args.width = int(args.w) * 3
    args.height = int(args.h) * 3

    
    try: 
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        run_simulation(args, client)
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')


if __name__ == '__main__':
    
    main()