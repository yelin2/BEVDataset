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

from PIL import Image
from PIL import ImageDraw

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

log_mutex = Lock()
v_concat = np.zeros((1200, 800, 3))

def metersetting(w, h, x_meter, y_meter, fov):
    '''
        w, h, x_meter, y_meter, fov -> cam height & meter per pixel
    '''
    fx = w /(2 * np.tan(fov * np.pi / 360))
    fy = h /(2 * np.tan(fov * np.pi / 360))
    cx = w/2
    cy = h/2

    x_rad = np.radians(fov/2)
    
    cam_height = x_meter/np.tan(x_rad)

    x_mpp = x_meter/w
    y_mpp = y_meter/h
    if w != h:
        y_meter = (x_meter/w) * h
        return cam_height/2, x_mpp, y_meter

    return cam_height/2, x_mpp, y_mpp

def carlaimg_to_np(carla_img):
    carla_img.convert(carla.ColorConverter.Raw)
    img_bgra = np.array(carla_img.raw_data).reshape((carla_img.height,carla_img.width,4))
    img_rgb = np.zeros((carla_img.height,carla_img.width,3))
    img_rgb[:,:,0] = img_bgra[:,:,2]
    img_rgb[:,:,1] = img_bgra[:,:,1]
    img_rgb[:,:,2] = img_bgra[:,:,0]
    img_rgb = np.uint8(img_rgb)
    image = Image.fromarray(img_rgb, 'RGB')
    return(np.array(image))



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
    #---------------cam setting----------------------------
    argparser.add_argument(
        '--w',
        metavar='W',
        default=480,
        type=int,
        help='cam setting(w)')
    argparser.add_argument(
        '--h',
        metavar='W',
        default=480,
        type=int,
        help='cam setting(h)')
    argparser.add_argument(
        '--x_meter',
        metavar='X',
        default=50,
        type=int,
        help='cam setting(x_meter)')
    argparser.add_argument(
        '--y_meter',
        metavar='Y',
        default=50,
        type=int,
        help='cam setting(y_meter)')
    argparser.add_argument(
        '--fov',
        metavar='F',
        default=70.4,
        type=int,
        help='cam setting(fov)')
    #---------------cam setting----------------------------
    

    args = argparser.parse_args()
    
    vehicles_list = []
    nonvehicles_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        world = client.get_world()

        #! layered map
        # world.unload_map_layer(carla.MapLayer.Foliage)
        # world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        # world.unload_map_layer(carla.MapLayer.Props)
        # world.unload_map_layer(carla.MapLayer.StreetLights)
        # world.unload_map_layer(carla.MapLayer.Decals)
        # world.unload_map_layer(carla.MapLayer.Buildings)
        # world.unload_map_layer(carla.MapLayer.Decals)

        # world.unload_map_layer(carla.MapLayer.Particles)
 
        print('\nRUNNING in synchronous mode\n')
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        ego_blueprints = world.get_blueprint_library().filter('vehicle.citroen.c3')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

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
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            spawn_points.pop(0)

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        print('Created %d npc vehicles \n' % len(vehicles_list))
        
        # -------------
        # Spawn Walkers
        # -------------
        # blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*', '2')
        blueprintsWalkers = world.get_blueprint_library().filter('walker.*')

        walkers_list = []


        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if 0:
            world.set_pedestrians_seed(0)
            random.seed(0)
        # 1. take all the random locations to spawn
        spawn_points_w = []
        for i in range(100):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points_w.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points_w:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2


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

        # Spawn ego vehicle
        ego_bp = random.choice(ego_blueprints)
        ego_transform = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        vehicles_list.append(ego_vehicle)
        ego_vehicle.set_autopilot(True)
        print('Ego-vehicle ready')

        # -----------------------------
        # Spawn sensors
        # -----------------------------

        #---------------------target-----------------------------

        # Spawn target RGB camera
        height, x_m_per_pxl, y_m_per_pxl = metersetting(args.w, args.h, args.x_meter, args.y_meter, args.fov)
        print("meter per pixel: ", x_m_per_pxl, y_m_per_pxl)
        t_sem_location = carla.Location(0,0,height)     #! BEV transform
        t_sem_rotation = carla.Rotation(-90, 0, 0)
        cam_transform = carla.Transform(t_sem_location, t_sem_rotation)
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        cam_bp.set_attribute('image_size_x', str(args.w))
        cam_bp.set_attribute('image_size_y', str(args.h))
        cam_bp.set_attribute('fov', str(args.fov))
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(cam)
        cam_queue = queue.Queue()
        cam.listen(cam_queue.put)
        q_list.append(cam_queue)
        cam_idx = idx
        idx = idx+1
        print('RGB camera ready')

        # Spawn target depth camera
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('sensor_tick', str(tick_sensor))
        depth_bp.set_attribute('image_size_x', str(args.w))
        depth_bp.set_attribute('image_size_y', str(args.h))
        depth_bp.set_attribute('fov', str(args.fov))
        depth = world.spawn_actor(depth_bp, cam_transform, attach_to=ego_vehicle)
        cc_depth_log = carla.ColorConverter.LogarithmicDepth
        nonvehicles_list.append(depth)
        depth_queue = queue.Queue()
        depth.listen(depth_queue.put)
        q_list.append(depth_queue)
        depth_idx = idx
        idx = idx+1
        print('Depth camera ready')

        # Spawn target segmentation camera
        # if save_segm:
        segm_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        segm_bp.set_attribute('sensor_tick', str(tick_sensor))
        segm_bp.set_attribute('image_size_x', str(args.w))
        segm_bp.set_attribute('image_size_y', str(args.h))
        segm_bp.set_attribute('fov', str(args.fov))
        segm = world.spawn_actor(segm_bp, cam_transform, attach_to=ego_vehicle)
        cc_segm = carla.ColorConverter.CityScapesPalette
        nonvehicles_list.append(segm)
        segm_queue = queue.Queue()
        segm.listen(segm_queue.put)
        q_list.append(segm_queue)
        segm_idx = idx
        idx = idx+1
        print('Segmentation camera ready')
        #--------------------------------------------------
        

        #-------front-------#
        car_cam_fov = '70'
        # Spawn target RGB camera
        f_sem_location = carla.Location(1, 0.0159456324149, 1.65)     #! BEV transform
        f_sem_rotation = carla.Rotation(0.04612719483860205, -(-90.32322642770004 + 90), -90.32571568590001+ 90)
        f_cam_transform = carla.Transform(f_sem_location, f_sem_rotation)
        f_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        f_cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        f_cam_bp.set_attribute('image_size_x', str(args.w))
        f_cam_bp.set_attribute('image_size_y', str(args.h))
        f_cam_bp.set_attribute('fov', car_cam_fov)
        f_cam = world.spawn_actor(f_cam_bp, f_cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(f_cam)
        f_cam_queue = queue.Queue()
        f_cam.listen(f_cam_queue.put)
        q_list.append(f_cam_queue)
        f_cam_idx = idx
        idx = idx+1
        
        # # Spawn target depth camera
        # f_depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        # f_depth_bp.set_attribute('sensor_tick', str(tick_sensor))
        # f_depth_bp.set_attribute('image_size_x', str(args.w))
        # f_depth_bp.set_attribute('image_size_y', str(args.h))
        # f_depth_bp.set_attribute('fov', car_cam_fov)
        # f_depth = world.spawn_actor(f_depth_bp, f_cam_transform, attach_to=ego_vehicle)
        # nonvehicles_list.append(f_depth)
        # f_depth_queue = queue.Queue()
        # f_depth.listen(f_depth_queue.put)
        # q_list.append(f_depth_queue)
        # f_depth_idx = idx
        # idx = idx+1

        # # Spawn target segmentation camera
        # # if save_segm:
        # f_segm_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # f_segm_bp.set_attribute('sensor_tick', str(tick_sensor))
        # f_segm_bp.set_attribute('image_size_x', str(args.w))
        # f_segm_bp.set_attribute('image_size_y', str(args.h))
        # f_segm_bp.set_attribute('fov', car_cam_fov)
        # f_segm = world.spawn_actor(f_segm_bp, f_cam_transform, attach_to=ego_vehicle)
        # nonvehicles_list.append(f_segm)
        # f_segm_queue = queue.Queue()
        # f_segm.listen(f_segm_queue.put)
        # q_list.append(f_segm_queue)
        # f_segm_idx = idx
        # idx = idx+1
        #--------------------------#

        #-------front_right-------#
        # Spawn target RGB camera
        fr_sem_location = carla.Location(1, 0.493404796419, 1.65)     #! BEV transform
        fr_sem_rotation = carla.Rotation(0.5188438566960005, -(-146.40439790300002 + 90), -90.78202358850001 + 90)
        fr_cam_transform = carla.Transform(fr_sem_location, fr_sem_rotation)
        fr_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        fr_cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        fr_cam_bp.set_attribute('image_size_x', str(args.w))
        fr_cam_bp.set_attribute('image_size_y', str(args.h))
        fr_cam_bp.set_attribute('fov', car_cam_fov)
        fr_cam = world.spawn_actor(fr_cam_bp, fr_cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(fr_cam)
        fr_cam_queue = queue.Queue()
        fr_cam.listen(fr_cam_queue.put)
        q_list.append(fr_cam_queue)
        fr_cam_idx = idx
        idx = idx+1
        
        # # Spawn target depth camera
        # fr_depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        # fr_depth_bp.set_attribute('sensor_tick', str(tick_sensor))
        # fr_depth_bp.set_attribute('image_size_x', str(args.w))
        # fr_depth_bp.set_attribute('image_size_y', str(args.h))
        # fr_depth_bp.set_attribute('fov', car_cam_fov)
        # fr_depth = world.spawn_actor(fr_depth_bp, fr_cam_transform, attach_to=ego_vehicle)
        # nonvehicles_list.append(fr_depth)
        # fr_depth_queue = queue.Queue()
        # fr_depth.listen(fr_depth_queue.put)
        # q_list.append(fr_depth_queue)
        # fr_depth_idx = idx
        # idx = idx+1

        # # Spawn target segmentation camera
        # # if save_segm:
        # fr_segm_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # fr_segm_bp.set_attribute('sensor_tick', str(tick_sensor))
        # fr_segm_bp.set_attribute('image_size_x', str(args.w))
        # fr_segm_bp.set_attribute('image_size_y', str(args.h))
        # fr_segm_bp.set_attribute('fov', car_cam_fov)
        # fr_segm = world.spawn_actor(fr_segm_bp, fr_cam_transform, attach_to=ego_vehicle)
        # nonvehicles_list.append(fr_segm)
        # fr_segm_queue = queue.Queue()
        # fr_segm.listen(fr_segm_queue.put)
        # q_list.append(fr_segm_queue)
        # fr_segm_idx = idx
        # idx = idx+1
        #--------------------------#

        #-------front_left-------#
        # Spawn target RGB camera
        fl_sem_location = carla.Location(1, -0.494631336551, 1.65)     #! BEV transform
        fl_sem_rotation = carla.Rotation(0.12143609391200118, -(-34.8390355956 + 90), -89.85977500319999 + 90)
        #pitch roll yaw
        fl_cam_transform = carla.Transform(fl_sem_location, fl_sem_rotation)
        fl_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        fl_cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        fl_cam_bp.set_attribute('image_size_x', str(args.w))
        fl_cam_bp.set_attribute('image_size_y', str(args.h))
        fl_cam_bp.set_attribute('fov', car_cam_fov)
        fl_cam = world.spawn_actor(fl_cam_bp, fl_cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(fl_cam)
        fl_cam_queue = queue.Queue()
        fl_cam.listen(fl_cam_queue.put)
        q_list.append(fl_cam_queue)
        fl_cam_idx = idx
        idx = idx+1
        
        # # Spawn target depth camera
        # b_depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        # b_depth_bp.set_attribute('sensor_tick', str(tick_sensor))
        # b_depth_bp.set_attribute('image_size_x', str(args.w))
        # b_depth_bp.set_attribute('image_size_y', str(args.h))
        # b_depth_bp.set_attribute('fov', car_cam_fov)
        # b_depth = world.spawn_actor(b_depth_bp, fl_cam_transform, attach_to=ego_vehicle)
        # nonvehicles_list.append(b_depth)
        # b_depth_queue = queue.Queue()
        # b_depth.listen(b_depth_queue.put)
        # q_list.append(b_depth_queue)
        # b_depth_idx = idx
        # idx = idx+1

        # # Spawn target segmentation camera
        # # if save_segm:
        # b_segm_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # b_segm_bp.set_attribute('sensor_tick', str(tick_sensor))
        # b_segm_bp.set_attribute('image_size_x', str(args.w))
        # b_segm_bp.set_attribute('image_size_y', str(args.h))
        # b_segm_bp.set_attribute('fov', car_cam_fov)
        # b_segm = world.spawn_actor(b_segm_bp, fl_cam_transform, attach_to=ego_vehicle)
        # nonvehicles_list.append(b_segm)
        # b_segm_queue = queue.Queue()
        # b_segm.listen(b_segm_queue.put)
        # q_list.append(b_segm_queue)
        # b_segm_idx = idx
        # idx = idx+1
        #--------------------------#

        #-------back_right-------#
        br_sem_location = carla.Location(0, 0.480568219723, 1.65)     #! BEV transform
        br_sem_rotation = carla.Rotation( 0.6190947610589997, -(159.200715506 + 90), -90.93206677999999 + 90)#pitch yaw roll
        br_cam_transform = carla.Transform(br_sem_location, br_sem_rotation)
        br_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        br_cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        br_cam_bp.set_attribute('image_size_x', str(args.w))
        br_cam_bp.set_attribute('image_size_y', str(args.h))
        br_cam_bp.set_attribute('fov', car_cam_fov)
        br_cam = world.spawn_actor(br_cam_bp, br_cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(br_cam)
        br_cam_queue = queue.Queue()
        br_cam.listen(br_cam_queue.put)
        q_list.append(br_cam_queue)
        br_cam_idx = idx
        idx = idx+1

        
        #--------------------------#

        #-------back_left-------#
        bl_sem_location = carla.Location(0, 0.484795032713, 1.65)     #! BEV transform
        bl_sem_rotation = carla.Rotation(-0.21518275753700122, -(18.600246142799996 + 90), -90.91736319750001 + 90)#pitch roll yaw
        bl_cam_transform = carla.Transform(bl_sem_location, bl_sem_rotation)
        bl_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bl_cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        bl_cam_bp.set_attribute('image_size_x', str(args.w))
        bl_cam_bp.set_attribute('image_size_y', str(args.h))
        bl_cam_bp.set_attribute('fov', car_cam_fov)
        bl_cam = world.spawn_actor(bl_cam_bp, bl_cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(bl_cam)
        bl_cam_queue = queue.Queue()
        bl_cam.listen(bl_cam_queue.put)
        q_list.append(bl_cam_queue)
        bl_cam_idx = idx
        idx = idx+1
        
        #--------------------------#

        #-------back-------#

        b_sem_location = carla.Location(-1.5, 0.00345136761476, 1.65)     #! BEV transform
        b_sem_rotation = carla.Rotation(0.22919685786400154, -(89.86124500000001 + 90), -89.0405962694 + 90)#pitch yaw roll
        b_cam_transform = carla.Transform(b_sem_location, b_sem_rotation)
        b_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        b_cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        b_cam_bp.set_attribute('image_size_x', str(args.w))
        b_cam_bp.set_attribute('image_size_y', str(args.h))
        b_cam_bp.set_attribute('fov', '110')
        b_cam = world.spawn_actor(b_cam_bp, b_cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(b_cam)
        b_cam_queue = queue.Queue()
        b_cam.listen(b_cam_queue.put)
        q_list.append(b_cam_queue)
        b_cam_idx = idx
        idx = idx+1
        
        #--------------------------#



        # Begin the loop
        time_sim = 0
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
                walkers_raw = world.get_actors().filter('walker.*')
                vehicles_raw = world.get_actors().filter('vehicle.*')
                snap = data[tick_idx]
                rgb_img = data[cam_idx]
                depth_img = data[depth_idx]
                segm_img = data[segm_idx]
                f_rgb = data[f_cam_idx]
                fr_rgb = data[fr_cam_idx]
                fl_rgb = data[fl_cam_idx]
                b_rgb = data[b_cam_idx]
                br_rgb = data[br_cam_idx]
                bl_rgb = data[bl_cam_idx]
                
                # Attach additional information to the snapshot
                walkers = cva.snap_processing(walkers_raw, snap)
                vehicles = cva.snap_processing(vehicles_raw, snap)

                # Save depth image, RGB image, and Bounding Boxes data
                if save_depth:
                    depth_img.save_to_disk('out_depth/%06d.png' % depth_img.frame, cc_depth_log)
                depth_meter = cva.extract_depth(depth_img)
                walker_filtered, walker_removed =  cva.auto_annotate(walkers, cam, depth_meter, json_path='vehicle_class_json_file.txt')
                walker_box_rgb = cva.save_output(rgb_img, walker_filtered['bbox'], walker_filtered['class'], walker_removed['bbox'], walker_removed['class'], save_patched=True, out_format='json')
                

                vehicle_filtered, vehicle_removed =  cva.auto_annotate(vehicles, cam, depth_meter, json_path='vehicle_class_json_file.txt')
                vehicle_box_rgb = cva.save_output(walker_box_rgb, vehicle_filtered['bbox'], vehicle_filtered['class'], \
                    vehicle_removed['bbox'], vehicle_removed['class'], save_patched=True, out_format='json', second = True, for_vehicle_img = rgb_img)
                
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


                #--------------------------------------------------
                global log_mutex
                global v_concat
                global car_camera

                log_mutex.acquire()
                v_concat = cv2.hconcat([vehicle_box_rgb, lane_road_seg])
                f_camera = cv2.hconcat([carlaimg_to_np(f_rgb),carlaimg_to_np(fr_rgb),carlaimg_to_np(fl_rgb) ])
                b_camera = cv2.hconcat([carlaimg_to_np(b_rgb),carlaimg_to_np(br_rgb),carlaimg_to_np(bl_rgb) ])
                car_camera = cv2.vconcat([f_camera, b_camera])
                print('main', v_concat.shape)
                log_mutex.release()
                
                
                # cv2.imshow('detection & segmentation', v_concat)

                
                
                # Uncomment if you want to save the data in darknet format
                #cva.save2darknet(filtered['bbox'], filtered['class'], rgb_img)

                # Save segmentation image
                if save_segm:
                    segm_img = data[segm_idx]
                    segm_img.save_to_disk('out_segm/%06d.png' % segm_img.frame, cc_segm)

                
                time_sim = 0
            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        cva.save2darknet(None,None,None,save_train=True)
        try:
            cam.stop()
            depth.stop()
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

def visualize():
    while True:
        global log_mutex
        global v_concat
        global car_camera


        log_mutex.acquire()
        # print('vis', v_concat.shape)
        cv2.imshow('detection & segmentation', v_concat)
        cv2.imshow('car camera', car_camera)
        log_mutex.release()
        cv2.waitKey(10)


# if __name__ == '__main__':

v_concat = np.zeros((1200, 800, 3))
car_camera = np.zeros((1200, 800, 3))

try:
    main_ = Thread(target=main)
    main_.start()
    # main_.join()


    vis = Thread(target=visualize)
    vis.start()
    # vis.join()
    # main()
except KeyboardInterrupt:
    pass
finally:
    print('\ndone.')