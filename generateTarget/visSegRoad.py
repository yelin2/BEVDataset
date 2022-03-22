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

import pygame
from PIL import Image


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

def metersetting(w, h, x_meter,  fov):
    '''
        w, h, x_meter, fov -> cam height & meter per pixel calculate
    '''
    fx = w /(2 * np.tan(fov * np.pi / 360))
    fy = h /(2 * np.tan(fov * np.pi / 360))
    cx = w/2
    cy = h/2

    x_rad = np.radians(fov/2)
    x_meter = x_meter/2
    cam_height = x_meter/np.tan(x_rad)

    x_mpp = x_meter/w
    y_meter = x_mpp * h
    return cam_height, x_mpp, y_meter

def main():

    # -----------------------------
    #         pygame setting
    # -----------------------------
    pygame.init()
    screen = pygame.display.set_mode((1700,700)) # w, h
    pygame.display.set_caption('vehicle segmentation')
    done = True
    clock = pygame.time.Clock()


    # -----------------------------
    #           argument
    # -----------------------------
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

    args = argparser.parse_args()
    
    vehicles_list = []
    nonvehicles_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

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
        # Spawn ego vehicle
        # -----------------------------
        q_list = []
        idx = 0
        
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx+1

        # Spawn ego vehicle
        ego_blueprints = world.get_blueprint_library().filter('vehicle.citroen.c3')
        ego_blueprints_num = random.randint(0, len(ego_blueprints)-1)
        ego_bp = ego_blueprints[ego_blueprints_num]
        # ego_bp = random.choice(blueprints)
        ego_transform = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        vehicles_list.append(ego_vehicle)
        ego_vehicle.set_autopilot(True)
        print('Ego-vehicle ready')


        # -----------------------------
        # Spawn BEV Seg Camera
        # -----------------------------
        bev_resolution = 512
        bev_meter = 25
        fov = 70
        height, x_m_per_pxl, y_meter = metersetting(bev_resolution, 
                                                    bev_resolution, 
                                                    bev_meter, fov)

        t_sem_location = carla.Location(bev_meter/2+1,0,height)     #! BEV transform
        t_sem_rotation = carla.Rotation(-90, 0, 0)
        cam_transform = carla.Transform(t_sem_location, t_sem_rotation)

        segm_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        segm_bp.set_attribute('sensor_tick', str(tick_sensor))
        segm_bp.set_attribute('image_size_x', str(bev_resolution))
        segm_bp.set_attribute('image_size_y', str(bev_resolution))
        segm_bp.set_attribute('fov', str(fov))
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
        # Spawn Front RGB Camera
        # -----------------------------
        f_location = carla.Location(1, 0.0159456324149, 1.65)
        f_rotation = carla.Rotation(0.04612719483860205, -(-90.32322642770004 + 90), -90.32571568590001+ 90)
        fcam_transform = carla.Transform(f_location, f_rotation)
        fcam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        fcam_bp.set_attribute('sensor_tick', str(tick_sensor))
        fcam_bp.set_attribute('image_size_x', str(512))
        fcam_bp.set_attribute('image_size_y', str(512))
        fcam_bp.set_attribute('fov', str(70))
        fcam = world.spawn_actor(fcam_bp, fcam_transform, attach_to=ego_vehicle)
        
        nonvehicles_list.append(fcam)
        fcam_queue = queue.Queue()
        fcam.listen(fcam_queue.put)
        q_list.append(fcam_queue)
        fcam_idx = idx
        idx = idx+1
        print('front camera ready')


        # -----------------------------
        # Spawn Front Segmentation Camera
        # -----------------------------
        fseg_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        fseg_bp.set_attribute('sensor_tick', str(tick_sensor))
        fseg_bp.set_attribute('image_size_x', str(512))
        fseg_bp.set_attribute('image_size_y', str(512))
        fseg_bp.set_attribute('fov', str(70))
        fseg = world.spawn_actor(fseg_bp, fcam_transform, attach_to=ego_vehicle)
        
        nonvehicles_list.append(fseg)
        fseg_queue = queue.Queue()
        fseg.listen(fseg_queue.put)
        q_list.append(fseg_queue)
        fseg_idx = idx
        idx = idx+1
        print('front seg camera ready')



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
                #         Get front rgb Image
                # -----------------------------
                frgb_img = data[fcam_idx]

                H, W = frgb_img.height, frgb_img.width

                np_frgb = np.frombuffer(frgb_img.raw_data, dtype=np.dtype("uint8")) 
                np_frgb = np.reshape(np_frgb, (H, W, 4)) # RGBA format
                np_frgb = np_frgb[:, :, :3] #  Take only RGB


                # -----------------------------
                #         Get front rgb Image
                # -----------------------------
                fseg_img = data[fseg_idx]

                H, W = fseg_img.height, fseg_img.width

                np_fseg = np.frombuffer(fseg_img.raw_data, dtype=np.dtype("uint8")) 
                np_fseg = np.reshape(np_fseg, (H, W, 4)) # RGBA format
                np_fseg = np_fseg[:, :, :3] #  Take only RGB

                #! TODO: np_fseg[:,:,2] convert to color


                # -----------------------------
                #        Get BEV Segmentation
                # -----------------------------
                segm_img = data[segm_idx]

                H, W = segm_img.height, segm_img.width

                np_seg = np.frombuffer(segm_img.raw_data, dtype=np.dtype("uint8")) 
                np_seg = np.reshape(np_seg, (H, W, 4)) # RGBA format
                np_seg = np_seg[:, :, :3] #  Take only RGB


                # initialize lane, road segmentation
                lane_road_seg = np.zeros((H, W, 3), dtype=np.uint8)


                # get lane, road mask
                lane_mask = (np_seg[:,:,2] == 6)
                road_mask = (np_seg[:,:,2] == 7)

                # 18: trafficLight, 5: Pole, 12: trafficSign
                # traffic_light_mask = np.logical_or((np_seg[:,:,2]==18), (np_seg[:,:,2]==5))     
                # traffic_light_mask = np.logical_or(traffic_light_mask, (np_seg[:,:,2]==12))

                # create lane_seg, road_seg
                lane_road_seg[lane_mask, :] = (255, 255, 255)
                lane_road_seg[road_mask, :] = (114, 114, 114)
                # lane_road_seg[traffic_light_mask, :] = (100, 100, 100)



                # -----------------------------
                #        Concat RGB, Seg
                # -----------------------------
                v_concat = cv2.hconcat([np_frgb, lane_road_seg])

                # save target
                import matplotlib.pyplot as plt
                filename = '{0:010d}'.format(cnt)
                plt.imsave(f'target/frgb/{filename}.png', np_frgb)
                fseg_img.save_to_disk(f'target/fseg/{filename}.png', carla.ColorConverter.CityScapesPalette)
                # plt.imsave(f'target/fseg/{filename}.png', lane_road_seg[:,:,0])
                plt.imsave(f'target/bev/{filename}.png', lane_road_seg[:,:,0], cmap='gray')
                cnt = cnt+1

                # * Pygame plot * #
                opencv_image = v_concat[:,:,::-1]  #Since OpenCV is BGR and pygame is RGB, it is necessary to convert it.
                shape = opencv_image.shape[1::-1]  #OpenCV(height,width,Number of colors), Pygame(width, height)So this is also converted.
                pygame_image = pygame.image.frombuffer(opencv_image.tostring(), shape, 'RGB')

                screen.blit(pygame_image, (0,0)) # put in the screen

                pygame.display.update() #update
                clock.tick(60)
                
                time_sim = 0

            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        try:
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
        print('\ndone')