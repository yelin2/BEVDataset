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
import json
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

        # print(client.get_available_maps())
        # world = client.load_world('Town05')      #! if you want change maps, change This one
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


        # sync mode + fixed delta time with 100Hz
        print('\nRUNNING in synchronous mode\n')
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01
        world.apply_settings(settings)

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
        front_cam_fov = str(140)
        target_cam_fov = str(80)

        # target camera's transform
        height, x_m_per_pxl, y_meter = metersetting(args.w, args.h, args.x_meter, int(target_cam_fov))
        # print("height:", height,"meter per pixel:", x_m_per_pxl, "y_meter:", y_meter*2)
        BEV_transform = carla.Transform(carla.Location(args.x_meter/2+1, 0, height), 
                                        carla.Rotation(-90, 0, 0))
        Front_transform = carla.Transform(carla.Location(1, 0.0159456324149, 1.65), 
                                          carla.Rotation(0.04612719483860205, -(-90.32322642770004 + 90), -90.32571568590001+ 90))
        Kinematic_transform = carla.Transform(carla.Location(0, 0, 0), 
                                              carla.Rotation(0, 0, 0))

        #? tick_sensor = call callback function rate after world.tick() in simulation
        #? to synchronize, need to set this value = 0
        tick_sensor = 0
        
        
        # create target camera sensor 
        #! TODO: Decide bevseg resolution
        front_h, front_w = 512, 512
        fseg = SensorManager(world, display_manager, 'SEMANTICCamera', Front_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(front_w),
                'image_size_y': str(front_h), 'fov': front_cam_fov}, display_pos=[1, 0], show = args.show) # target depth

        frgb = SensorManager(world, display_manager, 'RGBCamera', Front_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(front_w),
                'image_size_y': str(front_h), 'fov': front_cam_fov}, display_pos=[1, 0], show = args.show) # target depth

        bevseg = SensorManager(world, display_manager, 'SEMANTICCamera', BEV_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': target_cam_fov}, display_pos=[1, 0], show = args.show) # target depth
        
        gps = SensorManager(world, display_manager, 'GNSS', Kinematic_transform, ego_vehicle)

        imu = SensorManager(world, display_manager, 'IMU', Kinematic_transform, ego_vehicle)

       # cam manage q list append
        q_list.append(fseg.q)
        fseg_idx = idx
        idx = idx+1
        q_list.append(frgb.q)
        frgb_idx = idx
        idx = idx+1
        q_list.append(bevseg.q)
        bevseg_idx = idx
        idx = idx+1
        q_list.append(gps.q)
        gps_idx = idx
        idx = idx+1
        q_list.append(imu.q)
        imu_idx = idx
        idx = idx+1
        print("sensor setting done")



        # ===============================================================================
        #                                   Begin the loop
        # ===============================================================================
        cnt = 0
        time_sim = 0
        timestamps = []
        gpss, imus = [], []
        call_exit = False


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
                
                cnt = cnt + 1
                
                if cnt % 51 != 0:       #! current save rate == 2Hz
                    continue

                # -----------------------------
                #         Get front rgb Image
                # -----------------------------
                frgb_img = data[frgb_idx]

                H, W = frgb_img.height, frgb_img.width

                np_frgb = np.frombuffer(frgb_img.raw_data, dtype=np.dtype("uint8")) 
                np_frgb = np.reshape(np_frgb, (H, W, 4))                # RGBA format
                np_frgb = cv2.cvtColor(np_frgb, cv2.COLOR_RGBA2BGR)     # BGR format to OpenCV

                # -----------------------------
                #         Get front segmentation Image
                # -----------------------------
                fseg_img = data[fseg_idx]


                # -----------------------------
                #        Get BEV Segmentation
                # -----------------------------
                segm_img = data[bevseg_idx]

                H, W = segm_img.height, segm_img.width

                np_seg = np.frombuffer(segm_img.raw_data, dtype=np.dtype("uint8")) 
                np_seg = np.reshape(np_seg, (H, W, 4)) # RGBA format
                np_seg = np_seg[:, 128:384, :3]        # Take only RGB   #! 60m x 30m 128:384 / 60m x 15m 192:320


                # initialize lane, road segmentation
                lane_road_seg = np.zeros((512, 256, 3), dtype=np.uint8)     #! 30m 128/ 60m 256


                # get lane, road mask
                lane_mask = (np_seg[:,:,2] == 6)
                road_mask = (np_seg[:,:,2] == 7)

                # 18: trafficLight, 5: Pole, 12: trafficSign
                # traffic_light_mask = np.logical_or((np_seg[:,:,2]==18), (np_seg[:,:,2]==5))     
                # traffic_light_mask = np.logical_or(traffic_light_mask, (np_seg[:,:,2]==12))

                # create lane_seg, road_seg
                lane_road_seg[lane_mask, :] = (255, 255, 255)
                lane_road_seg[road_mask, :] = (85, 85, 85)
                # lane_road_seg[traffic_light_mask, :] = (100, 100, 100)


                # -----------------------------
                #         Get front rgb Image
                # -----------------------------
                gps_data = data[gps_idx]
                imu_data = data[imu_idx]

                gpss.append(gps_data)
                imus.append(imu_data)

                # -----------------------------
                #           Save TimeStamp
                # -----------------------------
                snapshot = world.get_snapshot()
                timestamps.append(snapshot.timestamp)
                print(f'saved {cnt} data: time[{round(snapshot.timestamp.elapsed_seconds, 4)}] delta[{round(snapshot.timestamp.delta_seconds, 4)}]')

                # -----------------------------
                #           Save Target
                # -----------------------------
                import matplotlib.pyplot as plt
                filename = '{0:010d}'.format(cnt)
                plt.imsave(f'target/frgb/{filename}.png', np_frgb)      # OpenCV is BGR
                fseg_img.save_to_disk(f'target/fseg/{filename}.png', carla.ColorConverter.CityScapesPalette)
                plt.imsave(f'target/bev/6030m512256p/{filename}.png', lane_road_seg[:,:,0], cmap='gray')

                # * Pygame plot * #
                seg = np.zeros_like(np_frgb)
                h, w, _ = lane_road_seg.shape
                seg[:h, :w, :] = lane_road_seg

                v_concat = cv2.hconcat([np_frgb, seg])
                opencv_image = v_concat            #Since OpenCV is BGR and pygame is RGB, it is necessary to convert it.
                shape = opencv_image.shape[1::-1]  #OpenCV(height,width,Number of colors), Pygame(width, height)So this is also converted.
                pygame_image = pygame.image.frombuffer(opencv_image.tostring(), shape, 'RGB')

                screen.blit(pygame_image, (0,0)) # put in the screen

                pygame.display.update() #update
                clock.tick(60)
                
                time_sim = 0

            time_sim = time_sim + settings.fixed_delta_seconds

    finally:

        save_kinematics(gpss, imus)
        try:
            if display_manager:
                display_manager.destroy()
        except:
            print("Simulation ended before sensors have been created")


        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        world.apply_settings(original_settings)


def save_kinematics(gpss, imus):

    gps_list, imu_list, time_list = [], [], []
    assert len(gpss) == len(imus)

    # convert traj, timestamp to list
    for i, (gps, imu) in enumerate(zip(gpss, imus)):
        gps_list.append({'frame': gps.frame,
                        'time': gps.timestamp,
                        'lat': gps.latitude,
                        'lon': gps.longitude,
                        'alt': gps.altitude,
                        'x':gps.transform.location.x,
                        'y':gps.transform.location.y,
                        'z':gps.transform.location.z,
                        'roll':gps.transform.rotation.roll,
                        'pitch':gps.transform.rotation.pitch,
                        'yaw':gps.transform.rotation.yaw})
        
        imu_list.append({'frame': imu.frame,
                        'time': imu.timestamp,
                        'acc_x': imu.accelerometer.x,
                        'acc_y': imu.accelerometer.y,
                        'acc_z': imu.accelerometer.z,
                        'gyro_x': imu.gyroscope.x,
                        'gyro_y': imu.gyroscope.y,
                        'gyro_z': imu.gyroscope.z,
                        'x':gps.transform.location.x,
                        'y':gps.transform.location.y,
                        'z':gps.transform.location.z,
                        'roll':gps.transform.rotation.roll,
                        'pitch':gps.transform.rotation.pitch,
                        'yaw':gps.transform.rotation.yaw})

        assert gps.timestamp == imu.timestamp
        time_list.append({'time': gps.timestamp})

    # determine save path
    path = 'target/kinematics'

    if not os.path.isdir(path):
        os.makedirs(path)


    # save trajectory, timesamps
    with open(path + '/gps.json', 'w') as js:
        json.dump(gps_list, js, indent=4)

    with open(path + '/imu.json', 'w') as js:
        json.dump(imu_list, js, indent=4)

    with open(path + '/timestamps.json', 'w') as js:
        json.dump(time_list, js, indent=4)

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