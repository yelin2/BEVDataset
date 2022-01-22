import numpy as np
import glob
import cv2
import torch
import random
from PIL import Image
import os

def prepare_boxes(origin_size, target, ratio, use_cocoApi=False):
    '''
    params:
        origin_size: (h,w)
        ratio: (rw, rh)
        pad: (dw, dh)
    
    Return 
        t(dict): target of current image
            boxes(n, 4): ordered coordinate of bbox (xyxy format)
            labels(n, 1): orderd class of bbox 
            image_id: image id of current image
    '''

    #% if using letterbox
    #! shape fixing
    target = np.array(target)
    boxes = target*ratio

    return boxes

def __getitem__(i, res, img_files):
        ratio = res
        ori_size = 964
        image_path = img_files[i]
        od_path = image_path.replace('out_rgb_bbox_vehicle','vehicle_bbox',1)
        od_path= od_path.replace('png','txt',1)
        image = cv2.imread(image_path)
        
        f = open(od_path, 'r', encoding='utf-8')
        data = f.read()
        f.close()
        data = eval(data)
            
        box = data['bboxes']
        r_box = data['removed_bboxes']

        # *  letter box  * #
        aug_size = (int(ori_size * ratio), int(ori_size * ratio))
        image = cv2.resize(image, aug_size, interpolation = cv2.INTER_LINEAR)
        # *  prepare bbox target  * #
        target = prepare_boxes((ori_size, ori_size), box, ratio=ratio) # target box type: xyxy
        r_target = prepare_boxes((ori_size, ori_size), r_box, ratio=ratio) # target box type: xyxy


        return target, r_target, image


