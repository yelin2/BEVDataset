import glob
import os
import sys

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.7-%s.egg' % (
        sys.version_info.major,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import json
import numpy as np
import cv2
from utils import carla_vehicle_BEV as cva



def save_rgb(rgbs, args, cnt):
    '''
    save surround view RGB images

    Inputs
        rgbs(list): list of carla.Image
    '''
    imgs_path = args.data_root + args.scene_num

    for i, rgb in enumerate(rgbs):
        
        # rgb image to numpy array
        H, W = rgb.height, rgb.width

        np_rgb = np.frombuffer(rgb.raw_data, dtype=np.dtype("uint8")) 
        np_rgb = np.reshape(np_rgb, (H, W, 4)) # RGBA format
        np_rgb = np_rgb[:, :, :3] #  Take only RGB


        # save rgb
        img_path = imgs_path + '/img{0:02d}'.format(i)
        
        if not os.path.isdir(img_path):
            os.makedirs(img_path)

        filename = '{0:010d}'.format(cnt)
        cv2.imwrite(img_path + f'/{filename}.png', np_rgb)
    
    

def save_seg(seg_raw, args, cnt, clss = None):

    '''
    save semantic class map to RGB image

    Input
        seg_raw(carla.Image): BEV Semantic Segmentation Map
        clss(list): class to create segmentation target
    
    mapping
        keys: class value in carla segmentation
        values: RGB value in BEV target
            (keys) |(value)
                5  | pole
                6  | lane
                7  | road
                10 | vehicles
                12 | traffic sign
                18 | traffic light
    '''
    assert clss is not None, 'check segmentation class!'

    mapping = {10: (200, 200, 200),
                6: (255, 255, 255),
                7: (114, 114, 114),
                18: (100, 100, 100), 
                5: (100, 100, 100),
                12: (100, 100, 100)
                }


    # segmentation to numpy array
    H_seg, W_seg = seg_raw.height, seg_raw.width
    seg_img = np.frombuffer(seg_raw.raw_data, dtype=np.dtype("uint8")) 
    seg_img = np.reshape(seg_img, (H_seg, W_seg, 4)) # RGBA format
    seg_img = seg_img[:, :, :3] #  Take only RGB


    # initialize target segmentation
    seg_tar = np.zeros((H_seg, W_seg, 3), dtype=np.uint8)


    # get mask & fill target segmentation
    for cls in clss:
        mask = (seg_img[:,:,2]==cls)
        seg_tar[mask, :] = mapping[cls]


    # save image
    s_or_d = 'static' if args.is_static else 'dynamic'
    seg_path = args.data_root + args.scene_num + '/segmentation/' + s_or_d

    if not os.path.isdir(seg_path):
        os.makedirs(seg_path)
    
    filename = '{0:010d}'.format(cnt)
    cv2.imwrite(seg_path + f'/{filename}.png', seg_tar)






def save_bbox_instanceSeg(seg_raw, filtered, removed, args, cnt):
    '''
    save bounding box & instance segmentation target in txt file

    Input:
        seg_raw(carla.Image)
        filtered(list of dict)
        removed(list of dict)
    
    Output:
        list of dictionary
            (keys) |(value)
             bbox  | bbox's left-top, right-bottom point
            class  | bbox's class
     segmentation  | instance segmentation pixels (x coord, y coord)
         image_id  | bbox's image id
    '''
    objs = {}
    if args.include_removed:
        objs['bbox'] = filtered['bbox'] + removed['bbox']
        objs['class'] = filtered['class'] + removed['class']
    else:
        objs['bbox'] = filtered['bbox']
        objs['class'] = filtered['class']


    segs = []

    # segmentation to numpy array
    H_seg, W_seg = seg_raw.height, seg_raw.width
    seg_img = np.frombuffer(seg_raw.raw_data, dtype=np.dtype("uint8")) 
    seg_img = np.reshape(seg_img, (H_seg, W_seg, 4)) # RGBA format
    seg_img = seg_img[:, :, :3] #  Take only RGB


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

    # save
    obj_path = args.data_root + args.scene_num + '/object_detection'
    
    if not os.path.isdir(obj_path):
        os.makedirs(obj_path)

    filename = '{0:010d}'.format(cnt)

    cva.save_obj_output(objs, 
                        path=obj_path, 
                        image_id=filename, 
                        out_format='json')


def save_trajectory(traj, timestamp, args):
    '''
    save trajectory to json file
    
    Input:
        traj(list of carla.Transform): saved trajectory
        timestamp(list of carla.Timestamps): saved simulation time
    
    Output:
        trajectory.json
        timestamps.json
    '''
    assert len(traj) == len(timestamp)

    traj_list = []
    time_list = []

    # convert traj, timestamp to list
    for i, (point, time) in enumerate(zip(traj, timestamp)):
        traj_list.append({'seq': i,
                    'time': time.elapsed_seconds,
                    'x':point.location.x,
                    'y':point.location.y,
                    'z':point.location.z,
                    'roll':point.rotation.roll,
                    'pitch':point.rotation.pitch,
                    'yaw':point.rotation.yaw})
        
        time_list.append({'time': time.elapsed_seconds,
                            'hz': time.delta_seconds})


    # determine save path
    path = args.data_root + args.scene_num

    if not os.path.isdir(path):
        os.makedirs(path)


    # save trajectory, timesamps
    with open(path + '/trajectory.json', 'w') as js:
        json.dump(traj_list, js, indent=4)

    with open(path + '/timestamps.json', 'w') as js:
        json.dump(time_list, js, indent=4)



def load_trajectory(json_path):
    '''
    load json trajectory file
    '''
    assert json_path is not None

    with open(json_path, 'r') as js:
        traj = json.load(js)

    return traj
