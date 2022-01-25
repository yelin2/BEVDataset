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


        # sync mode + fixed delta time with 100Hz
        print('\nRUNNING in synchronous mode\n')
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01
        world.apply_settings(settings)

        # blueprints = world.get_blueprint_library().filter('vehicle.*')
        # get ego vehicle's blue prints
        ego_blueprints = world.get_blueprint_library().filter('vehicle.citroen.c3')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print("vehicle_number_of_spawn_points:", number_of_spawn_points)


        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points
        

        # ===============================================================================
        #                           Spawn ego vehicle and sensors
        # ===============================================================================
        q_list = []
        idx = 0
        
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx+1

        # # Spawn ego vehicle
        ego_vehicle =None
        while not ego_vehicle:
            ego_blueprints_num = random.randint(0, len(ego_blueprints)-1)
            ego_random_num = random.randint(0, len(spawn_points)-1)
            ego_bp = ego_blueprints[ego_blueprints_num]
            ego_transform = spawn_points[ego_random_num]
            ego_vehicle = world.try_spawn_actor(ego_bp, ego_transform)
            vehicles_list.append(ego_vehicle)
            ego_vehicle.set_autopilot(True)
            print('Ego-vehicle ready')
            # ego_vehicle.set_simulate_physics(False) #? 

            # for point draw
            if args.debug:
                print("\nego vehicle spawn points", ego_transform)

                # world origin point draw
                draw_arrow(world)

                # ego point draw
                draw_points(world, ego_transform, color = [255,0,0], point_size = 0.1, time = 100, is_list = 0)


        traffic_manager.global_percentage_speed_difference(30.0)


        # ===============================================================================
        #                                   Spawn sensors
        # ===============================================================================

        display_manager = DisplayManager(grid_size=[3, 3], window_size=[1050, 1050])

        # fov
        target_cam_fov = str(args.fov)

        # target camera's transform
        height, x_m_per_pxl, y_meter = metersetting(args.w, args.h, args.x_meter, args.fov)
        print("height:", height,"meter per pixel:", x_m_per_pxl, "y_meter:", y_meter*2)
        t_transform = carla.Transform(carla.Location(0,0,height), carla.Rotation(-90, 0, 0))
        

        #? tick_sensor = call callback function rate after world.tick() in simulation
        #? to synchronize, need to set this value = 0
        tick_sensor = 0
        
        
        # create target camera sensor         
        t_sem = SensorManager(world, display_manager, 'SEMANTICCamera', t_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': target_cam_fov}, display_pos=[1, 0], show = args.show) # target depth
        t_rgb = SensorManager(world, display_manager, 'RGBCamera', t_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': target_cam_fov}, display_pos=[1, 1], show = args.show) # target rgb
        t_depth = SensorManager(world, display_manager, 'DEPTHCamera', t_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': target_cam_fov}, display_pos=[1, 2], show = args.show) # target segmentation
        

       # cam manage q list append
        q_list.append(t_depth.q)
        t_depth_idx = idx
        idx = idx+1
        q_list.append(t_rgb.q)
        t_rgb_idx = idx
        idx = idx+1
        q_list.append(t_sem.q)
        t_sem_idx = idx
        idx = idx+1
        print("camera setting done")


        # ===============================================================================
        #                                   Begin the loop
        # ===============================================================================
        cnt = 0
        call_exit = False


        # load trajectory
        json_path = args.data_root + args.scene_num + '/trajectory.json'
        traj = load_trajectory(json_path)

        # follow trajectory
        for waypoint in traj:

            # set ego-vehicle's waypoint
            location = carla.Location(waypoint['x'],
                                        waypoint['y'],
                                        waypoint['z'])

            rotation = carla.Rotation(waypoint['pitch'],
                                        waypoint['yaw'],
                                        waypoint['roll'])

            transform = carla.Transform(location, rotation)
            print(transform)
            ego_vehicle.set_transform(transform)

            # run simulation
            nowFrame = world.tick()

            # pygame rendering
            display_manager.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break
            
            # ===============================================================================
            #                                   Get Data
            # ===============================================================================

            # retrieve data from queue
            data = [retrieve_data(q,nowFrame) for q in q_list]
            assert all(x.frame == nowFrame for x in data if x is not None)

            # Skip if any sensor data is not available
            if None in data:
                continue

            # get data from data queue
            vehicles_raw = world.get_actors().filter('vehicle.*')
            snap = data[tick_idx]
            rgb_img = data[t_rgb_idx]
            depth_img = data[t_depth_idx]
            segm_img = data[t_sem_idx]


            # ===============================================================================
            #                   Save road, roadline, traffic light Seg
            # ===============================================================================

            # save vehicle semantic segmentation
            save_seg(segm_img, args, cnt, clss = [5, 6, 7, 12, 18])
            snapshot = world.get_snapshot()


            # for show & save object detection image
            if args.show:
                show_od_image(vehicles_raw, snap, depth_img, rgb_img, t_depth)

            print(f'saved {cnt} data: time[{round(snapshot.timestamp.elapsed_seconds, 4)}] delta[{round(snapshot.timestamp.delta_seconds, 4)}]')
            cnt = cnt + 1


    finally:

        try:
            if display_manager:
                display_manager.destroy()
        except:
            print("Simulation ended before sensors have been created")


        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        world.apply_settings(original_settings)



def main():

    
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