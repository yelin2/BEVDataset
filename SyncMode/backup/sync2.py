#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
from sqlite3 import Timestamp
import sys
import datetime

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.7-%s.egg' % (
        sys.version_info.major,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import argparse
import carla
import cv2
import random
import logging
import json
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

def save_trajectory(traj, timestamp, args, running_time):
    timestamp= str(timestamp)
    running_time = str(running_time)
    
    assert len(traj) == len(timestamp)
    traj_dict = {}
    traj_list = []

    
    traj_dict['running_time'] = running_time
    traj_dict['hz'] = args.hz

    for i, (point, time) in enumerate(zip(traj, timestamp)):
        traj_list.append({'seq': i,
                    'time': time,
                    'x':point.location.x,
                    'y':point.location.y,
                    'z':point.location.z,
                    'roll':point.rotation.roll,
                    'pitch':point.rotation.pitch,
                    'yaw':point.rotation.yaw})
    
    traj_dict['trajectory'] = traj_list

    print(f'total running time is {running_time}: saved {len(traj_list)} waypoints')
    with open(args.traj_save_path + 'trajectory.json', 'w') as js:
        json.dump(traj_dict, js, indent=4)


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


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        # self.delta_seconds = 2
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        # print(x.timestamp for x in data)

        assert all(x.frame == self.frame for x in data)
        # print('data', data)
        # data [<carla.libcarla.WorldSnapshot object at 0x7f3bd720a510>, 
        # <carla.libcarla.Image object at 0x7f3bd720a450>, 
        # <carla.libcarla.Image object at 0x7f3bd720a4b0>]
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
        # Queue's data == world's data frame? 

    def draw_waypoints(self, waypoints, road_id=None, life_time=50.0):
        for waypoint in waypoints:
            if(waypoint.road_id == road_id):
                self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                            color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                            persistent_lines=True)


def draw_image(surface, image, blend=False):
    # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # array = np.reshape(array, (image.height, image.width, 4))
    # array = array[:, :, :3]
    # array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def run_simulation(args, client):
    actor_list=[]
    vehicles_list = []
    nonvehicles_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()



    world = client.get_world()
    ######## var ################
    seed = 100
    number_of_vehicles = 50 
    vehicles_list = []
    walkers_list = []
    traj = []
    timestamp = []
    e=0
    s=0
    try:
        # * setup traffic manager * #
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        # if args.seed is not None:
        traffic_manager.set_random_device_seed(seed)
        random.seed(seed)

        world = client.get_world()
 
        print('\nRUNNING in synchronous mode\n')
        traffic_manager.set_synchronous_mode(True)
        ##########################################################
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        ego_blueprints = world.get_blueprint_library().filter('vehicle.citroen.c3')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print("vehicle_number_of_spawn_points:", number_of_spawn_points)


        

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor


        

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            # * Car Spec * #
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
            ###################
            # * set autopilot * #
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            spawn_points.pop(0)

        for response in client.apply_batch_sync(batch, False):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        print('Created %d npc vehicles \n' % len(vehicles_list))
        ###################################################################
        # -------------
        # Spawn Walkers
        # -------------
        # blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*', '2')
        # blueprintsWalkers = world.get_blueprint_library().filter('walker.*')

        # walkers_list = []


        # # some settings
        # percentagePedestriansRunning = 0.0      # how many pedestrians will run
        # percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # if seed:
        #     world.set_pedestrians_seed(seed)
        # # 1. take all the random locations to spawn
        # spawn_points_w = []
        # for i in range(args.number_of_walkers):
        #     spawn_point = carla.Transform()
        #     loc = world.get_random_location_from_navigation()
        #     if (loc != None):
        #         spawn_point.location = loc
        #         spawn_points_w.append(spawn_point)
        # # 2. we spawn the walker object
        # batch = []
        # walker_speed = []
        # for spawn_point in spawn_points_w:
        #     walker_bp_num = random.randint(0, len(blueprintsWalkers)-1)
        #     walker_bp = blueprintsWalkers[walker_bp_num]
        #     # set as not invincible
        #     if walker_bp.has_attribute('is_invincible'):
        #         walker_bp.set_attribute('is_invincible', 'false')
        #     # set the max speed
        #     if walker_bp.has_attribute('speed'):
        #         if (random.random() > percentagePedestriansRunning):
        #             # walking
        #             walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
        #         else:
        #             # running
        #             walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        #     else:
        #         print("Walker has no speed")
        #         walker_speed.append(0.0)
        #     batch.append(SpawnActor(walker_bp, spawn_point))
        # results = client.apply_batch_sync(batch, True)
        # walker_speed2 = []
        # for i in range(len(results)):
        #     if results[i].error:
        #         logging.error(results[i].error)
        #     else:
        #         walkers_list.append({"id": results[i].actor_id})
        #         walker_speed2.append(walker_speed[i])
        # walker_speed = walker_speed2
        #############################################################
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
        ego_blueprints_num = random.randint(0, len(ego_blueprints)-1)
        ego_random_num = random.randint(0, len(spawn_points)-1)
        ego_bp = ego_blueprints[ego_blueprints_num]
        ego_transform = spawn_points[ego_random_num]
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        vehicles_list.append(ego_vehicle)
        ego_vehicle.set_autopilot(True)
        print('Ego-vehicle ready')


        ######################################################################33


      
        # m = world.get_map()
        # start_pose = random.choice(m.get_spawn_points())
        # waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        # vehicle = world.spawn_actor(
        #     random.choice(blueprint_library.filter('vehicle.*')),
        #     start_pose)
        # actor_list.append(vehicle)
        # vehicle.set_simulate_physics(False) #? 

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(0,0,50), carla.Rotation(-90, 0, 0)),
            attach_to=ego_vehicle)
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(0,0,50), carla.Rotation(-90, 0, 0)),
            attach_to=ego_vehicle)
        actor_list.append(camera_semseg)
        
        

        # Create a synchronous mode context.
        # num =0
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=10) as sync_mode:
            s = world.get_snapshot().timestamp.elapsed_seconds
            while True:
                if should_quit():
                    return
                clock.tick()
                # image = image_queue.get()
                


                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                print('timestamp1 : ',snapshot.timestamp)
                traj.append(ego_vehicle.get_transform())
                timestamp.append(datetime.timedelta(seconds=int(snapshot.timestamp.elapsed_seconds)))

                
                # waypoint = random.choice(waypoint.next(1.5))
                # vehicle.set_transform(waypoint.transform)


                ################################
                # Seg_Road
                ################################
                H, W = image_rgb.height, image_rgb.width
                np_rgb = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8")) 
                np_rgb = np.reshape(np_rgb, (H, W, 4)) # RGBA format
                np_rgb = np_rgb[:, :, :3] #  Take only RGB

                                
                # -----------------------------
                #        Get Segmentation
                # -----------------------------
                H, W = image_semseg.height, image_semseg.width

                np_seg = np.frombuffer(image_semseg.raw_data, dtype=np.dtype("uint8")) 
                np_seg = np.reshape(np_seg, (H, W, 4)) # RGBA format
                np_seg = np_seg[:, :, :3] #  Take only RGB


                # initialize lane, road segmentation
                lane_road_seg = np.zeros((H, W, 3), dtype=np.uint8)


                # get lane, road, vehicle mask
                lane_mask = (np_seg[:,:,2] == 6)
                road_mask = (np_seg[:,:,2] == 7)

                # create lane_seg, road_seg
                lane_road_seg[lane_mask, :] = (255, 255, 255)
                lane_road_seg[road_mask, :] = (114, 114, 114)


                # -----------------------------
                #        Concat RGB, Seg
                # -----------------------------
             
                # v_concat = cv2.hconcat([np_rgb, lane_road_seg])
                


                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                print('Simulation time: % 12s' % datetime.timedelta(seconds=int(snapshot.timestamp.elapsed_seconds)))
                # print('Simulation time: % 12s' % datetime.timedelta(seconds=int(snapshot.timestamp.elapsed_seconds)))

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                # draw_image(display, image_semseg, blend=True)
                # draw_image(display, lane_road_seg, blend=True)
                draw_image(display, lane_road_seg)
                draw_image(display, np_rgb, blend=True)



                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
            print('dd')
            e = world.get_snapshot().timestamp.elapsed_seconds

    finally:

        print('destroying actors.')
        running_time=datetime.timedelta(seconds=e-s)
        save_trajectory(traj, timestamp, args, running_time)

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        print('destroying %d nonvehicles' % len(nonvehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in nonvehicles_list])

        print('destroying %d sensors' % len(actor_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


        pygame.quit()
        print('done.')

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
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
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot. This does not autocomplete the scenario')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='role name of ego vehicle to control (default: "hero")')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--keep_ego_vehicle',
        action='store_true',
        help='do not destroy ego vehicle on exit')
    argparser.add_argument(
        '-nov', '--number-of-vehicles',
        metavar='N',
        default=50,
        type=int,
        help='number of vehicles (default: 10)')
    
    #! need to set
    argparser.add_argument(
        '--save_traj',
        default=True,
        help='save trajectory (default: True)')
    argparser.add_argument(
        '--traj_save_path',
        default='',
        help='save trajectory path (default: '')')
    argparser.add_argument(
        '--hz',
        default='60',
        type=int,
        help='path save hz (default: '')')


    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    
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