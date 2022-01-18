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

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


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

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.q = queue.Queue()
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)
    

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
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
        
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        self.q.put(image)
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
    
    def save_depth_image(self, image):
        self.q.put(image)

        t_start = self.timer.time()

        image.convert(carla.ColorConverter.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
    
    def save_semantic_image(self, image):
        self.q.put(image)

        t_start = self.timer.time()

        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()

def run_simulation(args, client):
    vehicles_list = []
    nonvehicles_list = []
    try:

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        world = client.get_world()
 
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

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[3, 3], window_size=[args.width, args.height])

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
        
        
        height, x_m_per_pxl, y_m_per_pxl = metersetting(args.w, args.h, args.x_meter, args.y_meter, args.fov)
        print("meter per pixel: ", x_m_per_pxl, "y_meter: ", y_m_per_pxl)
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
        
        
        t_depth = SensorManager(world, display_manager, 'SEMANTICCamera', t_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': str(args.fov)}, display_pos=[1, 0]) # target depth
        t_rgb = SensorManager(world, display_manager, 'RGBCamera', t_transform, ego_vehicle, \
                {'sensor_tick': str(tick_sensor), 'image_size_x': str(args.w),
                'image_size_y': str(args.h), 'fov': str(args.fov)}, display_pos=[1, 1]) # target rgb
        t_sem = SensorManager(world, display_manager, 'DEPTHCamera', t_transform, ego_vehicle, \
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

        
        while True:
            # Extract the available data
            
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
                walkers = cva.snap_processing(walkers_raw, snap)
                vehicles = cva.snap_processing(vehicles_raw, snap)

                # Save depth image, RGB image, and Bounding Boxes data
                if save_depth:
                    
                    depth_img.save_to_disk('out_depth/%06d.png' % depth_img.frame, cc_depth_log)
                depth_meter = cva.extract_depth(depth_img)
                walker_filtered, walker_removed =  cva.auto_annotate(walkers, t_rgb.sensor, depth_meter, json_path='vehicle_class_json_file.txt')
                walker_box_rgb = cva.save_output(rgb_img, walker_filtered['bbox'], walker_filtered['class'], walker_removed['bbox'], walker_removed['class'], save_patched=True, out_format='json')
                

                vehicle_filtered, vehicle_removed =  cva.auto_annotate(vehicles, t_rgb.sensor, depth_meter, json_path='vehicle_class_json_file.txt')
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

                # Save segmentation image
                if save_segm:
                    segm_img = data[t_sem_idx]
                    cc_segm = carla.ColorConverter.CityScapesPalette
                    segm_img.save_to_disk('out_segm/%06d.png' % segm_img.frame, cc_segm)

                
                time_sim = 0
            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        cva.save2darknet(None,None,None,save_train=True)
        try:
            # cam.stop()
            # depth.stop()
            # if save_segm:
            #     segm.stop()
            if display_manager:
                display_manager.destroy()
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
        metavar='H',
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

    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    

    args = argparser.parse_args()
    args.width = int(args.w) * 3
    args.height = int(args.h) * 3

    
    try: 
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        run_simulation(args, client)
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

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

if __name__ == '__main__':
    
    main()