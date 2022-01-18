### Example program to save several sensor data including bounding box
### Sensors: RGB Camera (+BoundingBox), De[th Camera, Segmentation Camera, Lidar Camera
### By Mukhlas Adib
### 2020
### Last tested on CARLA 0.9.10.1

### CARLA Simulator is licensed under the terms of the MIT license
### For a copy, see <https://opensource.org/licenses/MIT>
### For more information about CARLA Simulator, visit https://carla.org/

from fileinput import filename
import glob
import os
import sys
import time

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
from threading import Thread, Lock


def retrieve_data(sensor_queue, frame, timeout=5):
    while True:
        try:
            data = sensor_queue.get(True,timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data

save_rgb = True
save_depth = False
save_segm = False
save_lidar = False
tick_sensor = 1


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
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
        '-n', '--number-of-vehicles',
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
    
    # save path
    argparser.add_argument(
        '-data_root', '--data_root',
        default=None,
        help='save data path root (defualt: None)')
    argparser.add_argument(
        '-scene_num', '--scene_num',
        default=None,
        help='driving scene number (defualt: None)')

    args = argparser.parse_args()
    
    vehicles_list = []
    nonvehicles_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    assert (args.data_root is not None), 'you need to specify data root path!'
    assert (args.scene_num is not None), 'you need to specify driving scene number!'

    try:
        # -----------------------------
        #           Set world
        # -----------------------------
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        world = client.get_world()


        # -----------------------------
        #       Set layered map
        # -----------------------------
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        world.unload_map_layer(carla.MapLayer.Props)
        world.unload_map_layer(carla.MapLayer.StreetLights)
        world.unload_map_layer(carla.MapLayer.Decals)
        world.unload_map_layer(carla.MapLayer.Foliage)

 
        # ----------------------------------------------------------
        #       Get blueprints & spawn points from CARLA map
        # ----------------------------------------------------------
        print('\nRUNNING in synchronous mode\n')
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        blueprints = world.get_blueprint_library().filter('vehicle.*')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points
        

        # -----------------------------
        # Spawn ego vehicle and cameras
        # -----------------------------
        q_list = []
        idx = 0
        
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx+1

        # Spawn ego vehicle
        ego_bp = random.choice(blueprints)
        ego_transform = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        vehicles_list.append(ego_vehicle)
        ego_vehicle.set_autopilot(True)
        print('Ego-vehicle ready')

        # Spawn RGB camera
        t_sem_location = carla.Location(0,0,50)     #! BEV transform
        t_sem_rotation = carla.Rotation(-90, 0, 0)
        cam_transform = carla.Transform(t_sem_location, t_sem_rotation)
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(cam)
        cam_queue = queue.Queue()
        cam.listen(cam_queue.put)
        q_list.append(cam_queue)
        cam_idx = idx
        idx = idx+1
        print('RGB camera ready')


        # Spawn segmentation camera
        segm_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        segm_bp.set_attribute('sensor_tick', str(tick_sensor))
        segm = world.spawn_actor(segm_bp, cam_transform, attach_to=ego_vehicle)
        cc_segm = carla.ColorConverter.CityScapesPalette
        nonvehicles_list.append(segm)
        segm_queue = queue.Queue()
        segm.listen(segm_queue.put)
        q_list.append(segm_queue)
        segm_idx = idx
        idx = idx+1
        print('Segmentation camera ready')



        # -----------------------------
        #       Begin the loop
        # -----------------------------
        time_sim = 0
        cnt = 0
        while True:
            # Extract the available data
            nowFrame = world.tick()

            # Check whether it's time to capture data
            if time_sim >= tick_sensor:
                data = [retrieve_data(q,nowFrame) for q in q_list]
                assert all(x.frame == nowFrame for x in data if x is not None)

                # Skip if any sensor data is not available
                if None in data:
                    continue

                
                # -----------------------------
                #         Get RGB Image
                # -----------------------------
                rgb_img = data[cam_idx]

                H, W = rgb_img.height, rgb_img.width

                np_rgb = np.frombuffer(rgb_img.raw_data, dtype=np.dtype("uint8")) 
                np_rgb = np.reshape(np_rgb, (H, W, 4)) # RGBA format
                np_rgb = np_rgb[:, :, :3] #  Take only RGB

                                
                # -----------------------------
                #        Get Segmentation
                # -----------------------------
                segm_img = data[segm_idx]

                H, W = segm_img.height, segm_img.width

                np_seg = np.frombuffer(segm_img.raw_data, dtype=np.dtype("uint8")) 
                np_seg = np.reshape(np_seg, (H, W, 4)) # RGBA format
                np_seg = np_seg[:, :, :3] #  Take only RGB


                # initialize lane, road segmentation
                lane_road_seg = np.zeros((H, W, 3), dtype=np.uint8)


                # get lane, road, vehicle mask
                lane_mask = (np_seg[:,:,2] == 6)
                road_mask = (np_seg[:,:,2] == 7)
                traffic_light_mask = np.logical_or((np_seg[:,:,2]==18), (np_seg[:,:,2]==5))
                traffic_light_mask = np.logical_or(traffic_light_mask, (np_seg[:,:,2]==12))


                # create lane_seg, road_seg
                lane_road_seg[lane_mask, :] = (255, 255, 255)
                lane_road_seg[road_mask, :] = (114, 114, 114)
                lane_road_seg[traffic_light_mask, :] = (100, 100, 100)


                # -----------------------------
                #     Save RGB, Segmentation
                # -----------------------------
                img_path = args.data_root + args.scene_num + '/image_01/'
                ann_path = args.data_root + args.scene_num + '/segmentation/static/'

                if not os.path.isdir(img_path):
                    os.makedirs(img_path)
                if not os.path.isdir(ann_path):
                    os.makedirs(ann_path)

                filename = '{0:010d}'.format(cnt)

                cv2.imwrite(ann_path + f'{filename}.png', lane_road_seg)
                cv2.imwrite(img_path + f'{filename}.png', np_rgb)
                
                print('saved!')
                time_sim = 0
                cnt = cnt + 1

            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        cva.save2darknet(None,None,None,save_train=True)
        try:
            cam.stop()
            if save_segm:
                segm.stop()
        except:
            print("Simulation ended before sensors have been created")
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        print('destroying %d nonvehicles' % len(nonvehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in nonvehicles_list])

        time.sleep(0.5)



if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
