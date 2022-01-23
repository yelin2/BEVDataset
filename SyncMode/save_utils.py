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


def save_rgb(rgbs, args, cnt):
    img_path = args.data_root + args.scene_num

    for i, rgb in enumerate(rgbs):
        
        # rgb image to numpy array
        H, W = rgb.height, rgb.width

        np_rgb = np.frombuffer(rgb.raw_data, dtype=np.dtype("uint8")) 
        np_rgb = np.reshape(np_rgb, (H, W, 4)) # RGBA format
        np_rgb = np_rgb[:, :, :3] #  Take only RGB


        # save rgb
        img_path = img_path + '{0:02d}'.format(i)
        
        if not os.path.isdir(img_path):
            os.makedirs(img_path)

        filename = '{0:010d}'.format(cnt)
        cv2.imwrite(img_path + f'{filename}.png', np_rgb)
    
    

def save_seg(seg_raw, args, cnt, clss = None):

    '''
    5: pole
    6: lane
    7: road
    10: vehicles
    12: traffic sign
    18: traffic light
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
    cv2.imwrite(seg_path + f'{filename}.png', seg_tar)






def save_bbox_instanceSeg(depth_img):
    depth_meter = cva.extract_depth(depth_img)

    vehicles_raw = world.get_actors().filter('vehicle.*')
    vehicles = cva.snap_processing(vehicles_raw, snap)

    vehicle_filtered, vehicle_removed =  cva.auto_annotate(vehicles, 
                                                            cam, 
                                                            depth_meter, 
                                                            cls='vehicle')

