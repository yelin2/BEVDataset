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
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False

        blueprints = world.get_blueprint_library().filter('vehicle.*')

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


        # ----------------------------
        # Spawn simulators vehicles
        # ----------------------------
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
        

        # --------------------------
        # Spawn Walkers
        # --------------------------
        blueprintsWalkers = world.get_blueprint_library().filter('walker.*')
        walkers_list = []

        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
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

        # Spawn depth camera
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('sensor_tick', str(tick_sensor))
        depth = world.spawn_actor(depth_bp, cam_transform, attach_to=ego_vehicle)
        cc_depth_log = carla.ColorConverter.LogarithmicDepth
        nonvehicles_list.append(depth)
        depth_queue = queue.Queue()
        depth.listen(depth_queue.put)
        q_list.append(depth_queue)
        depth_idx = idx
        idx = idx+1
        print('Depth camera ready')

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
                #           Get data
                # -----------------------------
                snap = data[tick_idx]
                depth_img = data[depth_idx]
                seg_img = data[segm_idx]

                # segmentation to numpy array
                H_seg, W_seg = seg_img.height, seg_img.width
                seg_img = np.frombuffer(seg_img.raw_data, dtype=np.dtype("uint8")) 
                seg_img = np.reshape(seg_img, (H_seg, W_seg, 4)) # RGBA format
                seg_img = seg_img[:, :, :3] #  Take only RGB

                
                # -----------------------------
                #        Get bounding box
                # -----------------------------

                depth_meter = cva.extract_depth(depth_img)
                
                walkers_raw = world.get_actors().filter('walker.*')
                walkers = cva.snap_processing(walkers_raw, snap)
                
                walker_filtered, walker_removed = cva.auto_annotate(walkers, 
                                                                    cam, 
                                                                    depth_meter, 
                                                                    cls='walker')
                
                vehicles_raw = world.get_actors().filter('vehicle.*')
                vehicles = cva.snap_processing(vehicles_raw, snap)

                vehicle_filtered, vehicle_removed =  cva.auto_annotate(vehicles, 
                                                                        cam, 
                                                                        depth_meter, 
                                                                        cls='vehicle')

                # -----------------------------
                #       Get instance Seg
                # -----------------------------
                objs = {}
                objs['bbox'] = walker_filtered['bbox'] \
                                + vehicle_filtered['bbox']\

                                # + vehicle_removed['bbox']\
                                # + walker_removed['bbox']

                objs['class'] = walker_filtered['class']\
                                + vehicle_filtered['class']\

                                # + vehicle_removed['class']\
                                # + walker_removed['class']

                segs = []

                for bbox, cls in zip(objs['bbox'], objs['class']):
                    
                    # get bbox's left-top, right-bottom
                    w1, h1 = int(bbox[0,0]), int(bbox[0,1])
                    w2, h2 = int(bbox[1,0]), int(bbox[1,1])

                    # get instance pixel
                    cls_value = 10 if cls == 'vehicle' else 4
                    ins_mask = seg_img[h1:h2, w1:w2, 2] == cls_value

                    pix_h, pix_w = np.where(ins_mask == True)

                    pix_h += h1
                    pix_w += w1

                    pix = np.stack([pix_h, pix_w], axis=0)

                    segs.append(pix)

                objs['segmentation'] = segs



                # -----------------------------
                #       Save bbox, instance
                # -----------------------------

                path = '/home/rml/ws/BEVDataset/generateTarget/annotations/ObjectDetection/drive_001/'
                image_id = '{0:010d}'.format(cnt)

                cva.save_output(objs, 
                                path=path, 
                                image_id=image_id, 
                                out_format='json')


                
                time_sim = 0
                cnt = cnt + 1

            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        cva.save2darknet(None,None,None,save_train=True)
        try:
            cam.stop()
            depth.stop()
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
