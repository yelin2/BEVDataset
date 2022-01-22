### Example program to save several sensor data including bounding box
### Sensors: RGB Camera (+BoundingBox), De[th Camera, Segmentation Camera, Lidar Camera
### By Mukhlas Adib
### 2020
### Last tested on CARLA 0.9.10.1

### CARLA Simulator is licensed under the terms of the MIT license
### For a copy, see <https://opensource.org/licenses/MIT>
### For more information about CARLA Simulator, visit https://carla.org/

from ast import While
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

import argparse
import logging
import random
import queue
import numpy as np
from matplotlib import pyplot as plt
import cv2
import carla_vehicle_BEV as cva
from PIL import Image
from PIL import ImageDraw
from threading import Thread, Lock

from utils import *
from manajors import SensorManager, DisplayManager
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

        
    



save_rgb = True
save_depth = False
save_segm = False
save_lidar = False
tick_sensor = 1

log_mutex = Lock()
v_concat = np.zeros((1200, 800, 3))

    
        




def run_simulation(args, client):
    vehicles_list = []

    try:
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)
            random.seed(args.seed)

        # world = client.load_world('Town02')
        world = client.get_world()
        original_settings = world.get_settings()

 
        print('\nRUNNING in synchronous mode\n')
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01
        world.apply_settings(settings)

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        ego_blueprints = world.get_blueprint_library().filter('vehicle.citroen.c3')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print("vehicle_number_of_spawn_points:", number_of_spawn_points)

        spawn_points = suffled(args, spawn_points, number_of_spawn_points)


        

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            bp_num = random.randint(0, len(blueprints)-1)
            blueprint = blueprints[bp_num]
            if blueprint.has_attribute('color'):
                color_num = random.randint(0, len(blueprint.get_attribute('color').recommended_values)-1)

                color = blueprint.get_attribute('color').recommended_values[color_num]
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id_num = random.randint(0, len(blueprint.get_attribute('driver_id').recommended_values)-1)
                driver_id = blueprint.get_attribute('driver_id').recommended_values[driver_id_num]
                blueprint.set_attribute('driver_id', driver_id)
            
            if args.debug:
                draw_spawnpoints(world, transform, is_list = 0)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            spawn_points.pop(0)
        # * Sync * #
        for response in client.apply_batch_sync(batch, False):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        print('Created %d npc vehicles \n' % len(vehicles_list))
        
        # -----------------------------
        # Spawn ego vehicle and sensors
        # -----------------------------
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


        traffic_manager.global_percentage_speed_difference(30.0)


        # -----------------------------
        # Spawn sensors
        # -----------------------------

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[3, 3], window_size=[1050, 1050])

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 

        #transform
        fl_transform = carla.Transform(carla.Location(1, -0.494631336551, 1.65),\
                    carla.Rotation(0.12143609391200118, -(-34.8390355956 + 90), -89.85977500319999 + 90))
        f_transform = carla.Transform(carla.Location(1, 0.0159456324149, 1.65),\
                    carla.Rotation(0.04612719483860205, -(-90.32322642770004 + 90), -90.32571568590001+ 90))
        fr_transform = carla.Transform(carla.Location(1, 0.493404796419, 1.65), \
                        carla.Rotation(0.5188438566960005, -(-146.40439790300002 + 90), -90.78202358850001 + 90))
        bl_transform = carla.Transform(carla.Location(0, -0.484795032713, 1.65), \
                        carla.Rotation(-0.21518275753700122, -(18.600246142799996 + 90), -90.91736319750001 + 90))
        b_transform = carla.Transform(carla.Location(-1.5, 0.00345136761476, 1.65) , \
                        carla.Rotation(0.22919685786400154, -(89.86124500000001 + 90), -89.0405962694 + 90))
        br_transform = carla.Transform(carla.Location(0, 0.480568219723, 1.65) , \
                        carla.Rotation( 0.6190947610589997, -(159.200715506 + 90), -90.93206677999999 + 90))
        
        
        height, x_m_per_pxl, y_meter = metersetting(args.w, args.h, args.x_meter, args.fov)
        print("height:", height,"meter per pixel:", x_m_per_pxl, "y_meter:", y_meter*2)
        t_transform = carla.Transform(carla.Location(0,0,height), carla.Rotation(-90, 0, 0))
        
        car_cam_fov = '70'
        fl_rgb = SensorManager(world, display_manager, 'RGBCamera', fl_transform, 
                      ego_vehicle, {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
            'image_size_y': str(args.h), 'fov': car_cam_fov}, display_pos=[0, 0]) #front left
        f_rgb = SensorManager(world, display_manager, 'RGBCamera', f_transform, 
                      ego_vehicle, {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
            'image_size_y': str(args.h), 'fov': car_cam_fov}, display_pos=[0, 1])#front
        fr_rgb = SensorManager(world, display_manager, 'RGBCamera',fr_transform,ego_vehicle, \
            {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
            'image_size_y': str(args.h), 'fov': car_cam_fov}, display_pos=[0, 2])#front right
        
        
        t_sem = SensorManager(world, display_manager, 'SEMANTICCamera', t_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': str(args.fov)}, display_pos=[1, 0]) # target depth
        t_rgb = SensorManager(world, display_manager, 'RGBCamera', t_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': str(args.fov)}, display_pos=[1, 1]) # target rgb
        t_depth = SensorManager(world, display_manager, 'DEPTHCamera', t_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': str(args.fov)}, display_pos=[1, 2]) # target segmentation
        
        bl_rgb = SensorManager(world, display_manager, 'RGBCamera',bl_transform,
                      ego_vehicle, {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
            'image_size_y': str(args.h), 'fov': car_cam_fov}, display_pos=[2, 0])# back left rgb
        b_rgb = SensorManager(world, display_manager, 'RGBCamera',b_transform,
                      ego_vehicle, {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
            'image_size_y': str(args.h), 'fov': '110'}, display_pos=[2, 1])# back rgb
        br_rgb = SensorManager(world, display_manager, 'RGBCamera',br_transform,
                      ego_vehicle, {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
            'image_size_y': str(args.h), 'fov': car_cam_fov}, display_pos=[2, 2])# back right rgb
        
        
        q_list.append(fr_rgb.q)
        fr_rgb_idx = idx
        idx = idx+1
        q_list.append(fl_rgb.q)
        fl_rgb_idx = idx
        idx = idx+1
        q_list.append(f_rgb.q)
        f_rgb_idx = idx
        idx = idx+1
        q_list.append(t_depth.q)
        t_depth_idx = idx
        idx = idx+1
        q_list.append(t_rgb.q)
        t_rgb_idx = idx
        idx = idx+1
        q_list.append(t_sem.q)
        t_sem_idx = idx
        idx = idx+1
        q_list.append(br_rgb.q)
        br_rgb_idx = idx
        idx = idx+1
        q_list.append(b_rgb.q)
        b_rgb_idx = idx
        idx = idx+1
        q_list.append(bl_rgb.q)
        bl_rgb_idx = idx
        idx = idx+1
        print("camera setting done")

        # Begin the loop
        time_sim = 0
        cc_depth_log = carla.ColorConverter.LogarithmicDepth
        call_exit = False

        # with CarlaSyncMode(world, fl_rgb.sensor, f_rgb.sensor, fr_rgb.sensor, t_sem.sensor ,t_rgb.sensor, t_depth.sensor 
        #         ,bl_rgb.sensor ,b_rgb.sensor, br_rgb.sensor, fps=30) as sync_mode:
        while True:

            # print('in while')
            # display_manager.clock.tick()
            nowFrame = world.tick()
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
            # print('in while9')
            
            # Check whether it's time to capture data
            if time_sim >= tick_sensor:
                print('in while1')

                data = [retrieve_data(q,nowFrame) for q in q_list]
                print(all(x.frame == nowFrame for x in data if x is not None))
                assert all(x.frame == nowFrame for x in data if x is not None)

                # Skip if any sensor data is not available
                if None in data:
                    continue
                vehicles_raw = world.get_actors().filter('vehicle.*')
                snap = data[tick_idx]
                rgb_img = data[t_rgb_idx]
                depth_img = data[t_depth_idx]
                segm_img = data[t_sem_idx]
                f_rgb = data[f_rgb_idx]
                fr_rgb = data[fr_rgb_idx]
                fl_rgb = data[fl_rgb_idx]
                b_rgb = data[b_rgb_idx]
                br_rgb = data[br_rgb_idx]
                bl_rgb = data[bl_rgb_idx]
                
                # Attach additional information to the snapshot
                vehicles = cva.snap_processing(vehicles_raw, snap)

                # Save depth image, RGB image, and Bounding Boxes data
                if save_depth:
                    depth_img.save_to_disk('out_depth/%06d.png' % depth_img.frame, cc_depth_log)
                depth_meter = cva.extract_depth(depth_img)
                # walker_filtered, walker_removed =  cva.auto_annotate(walkers, t_rgb.sensor, depth_meter, json_path='vehicle_class_json_file.txt')
                # walker_box_rgb = cva.save_output(rgb_img, walker_filtered['bbox'], walker_filtered['class'], walker_removed['bbox'], walker_removed['class'], save_patched=True, out_format='json')
                

                vehicle_filtered, vehicle_removed =  cva.auto_annotate(vehicles, t_rgb.sensor, depth_meter, json_path='vehicle_class_json_file.txt')
                vehicle_box_rgb = cva.save_output(rgb_img, vehicle_filtered['bbox'], vehicle_filtered['class'], \
                    vehicle_removed['bbox'], vehicle_removed['class'], save_patched=True, out_format='json')

                # pygame display
                if t_depth.display_man.render_enabled():
                    vehicle_box_rgb = cv2.resize(vehicle_box_rgb, (350, 350)) 
                    t_depth.surface = pygame.surfarray.make_surface(vehicle_box_rgb.swapaxes(0, 1))
                
                

                
                #-------------------Segmentation-------------------

                H, W = segm_img.height, segm_img.width

                np_seg = np.frombuffer(segm_img.raw_data, dtype=np.dtype("uint8")) 
                np_seg = np.reshape(np_seg, (H, W, 4)) # RGBA format
                np_seg = np_seg[:, :, :3] #  Take only RGB


                # initialize lane, road segmentation
                lane_road_seg = np.zeros((H, W, 3), dtype=np.uint8)


                # get lane, road mask
                lane_mask = (np_seg[:,:,2] == 6)
                road_mask = (np_seg[:,:,2] == 7)

                # create lane_seg
                lane_road_seg[lane_mask, :] = (255, 255, 255)

                # lane_seg
                # lane_im = Image.fromarray(lane_road_seg)

                # create road_seg
                lane_road_seg[road_mask, :] = (114, 114, 114)

                # Save segmentation image
                if save_segm:
                    segm_img = data[t_sem_idx]
                    cc_segm = carla.ColorConverter.CityScapesPalette
                    segm_img.save_to_disk('out_segm/%06d.png' % segm_img.frame, cc_segm)

                
                time_sim = 0
                snapshot = world.get_snapshot()
                print('timestamp: ',snapshot.timestamp)
            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        cva.save2darknet(None,None,None,save_train=True)
        try:
            if display_manager:
                display_manager.destroy()
        except:
            print("Simulation ended before sensors have been created")


        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        if display_manager:
            display_manager.destroy()

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