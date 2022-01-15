import os
import math
import random

from typing import Any, Callable, Optional, Tuple, List
from pycocotools.coco import COCO
import cv2

# import glob
import numpy as np

from operator import itemgetter


from PIL import Image
import torch
from torch.utils import data
import torch.distributed as dist
import torch.nn.functional as F
from . import augmentation as aug
import copy


#% background = 0
coco_mapping = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:17, 19:18,
    20:19, 21:20, 22:21, 23:22, 24:23, 25:24, 27:25, 28:26, 31:27, 32:28, 33:29, 34:30, 35:31, 36:32, 37:33, 38:34
    , 39:35, 40:36, 41:37, 42:38, 43:39, 44:40, 46:41, 47:42, 48:43, 49:44, 50:45, 51:46, 52:47, 53:48, 54:49, 55:50
    , 56:51, 57:52, 58:53, 59:54, 60:55, 61:56, 62:57, 63:58, 64:59, 65:60, 67:61, 70:62, 72:63, 73:64, 74:65, 75:66
    , 76:67, 77:68, 78:69, 79:70, 80:71, 81:72, 82:73, 84:74, 85:75, 86:76, 87:77, 88:78, 89:79, 90:80
}

class CocoDetection(data.Dataset):
    def __init__(self, cfg, split, augment=True, ignore_label=255):
        '''
        path:
            Image: coco/{train/val}2017/*.jpg
            ground truth: coco/annotations/instances_{train/val}2017.json
            root: ../coco
        '''
        #! cocoDataset --> coco
        self.root = '../coco/' # cfg.DATA.ROOT # '../coco/'
        self.split = split
        annFile= f'{self.root}annotations/instances_{self.split}2017.json'
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # *  normalization params  * #
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        self.normalize = aug.Compose([
            aug.ToTensor(),
            aug.Normalize(mean=mean, std=std)
        ])


    def __getitem__(self, index: int):
        image_id = self.ids[index]

        # *  load image & bbox target  * #
        image = self._load_image(image_id)
        ann = self._load_target(image_id)


        # *  create segmentation target  * #
        for ttt in ann:                                              #% convert labels 91 -> 80
            ttt['category_id'] = coco_mapping[ttt['category_id']]
        seg = self._load_segmentation(image_id, copy.deepcopy(ann))
        target = {'image_id': image_id, 'annotations': ann}        


        # self.visualize(image, seg, target['boxes'])

        # *  Convert  * #
        target['labels'] -=1
        image, seg, target = self.normalize(image, seg, target)


        assert image.shape[0] ==3,  f'check image {self.coco.loadImgs(image_id)[0]["file_name"]}'
        assert len(seg.shape) ==2,  f'check seg label shape'

        return image, seg, target

    

    def visualize(self, image, seg, bboxes):
        ''' 
        self.visualize(image, seg, target['boxes'])
        '''
        for bbox in list(bboxes):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)
            cv2.rectangle(seg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        cv2.imwrite('image.png', image)
        cv2.imwrite('seg.png', seg)


    def __len__(self):
        return len(self.ids)


    def _load_image(self, id: int):
        '''
        load Image with CV2 and resize image
        max(resized image's width, height) must be self.image_size
        '''
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.root, f'{self.split}2017', path)
        image = cv2.imread(path)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        assert image is not None, 'Image Not Found ' + path

        return image


    def _load_target(self, id: int):
        '''
        return bbox annotation image[id] with pycocotools
        Return
            coco_target: [{bbox1}, {bbox2}, ..., {bboxn}]
            keys: 
                segmentation: 
                area: bbox area
                image_id: image id
                bbox: bbox coordinate (xywh format)
                category_id: bbox's class
                id: 
        '''
        coco_target = copy.deepcopy(self.coco.loadAnns(self.coco.getAnnIds(id)))    # * need to deepcopy becuase of python's call by object
        return coco_target


    def _load_segmentation(self, id: int, anns):
        img_info = self.coco.loadImgs(id)[0]
        mask = np.zeros((img_info['height'],img_info['width']), dtype=np.uint8)

        for i in range(len(anns)):
            anns[i]['num_pixel']=self.coco.annToMask(anns[i]).sum()

        newlist = sorted(anns, key=itemgetter('num_pixel'), reverse=True)

        for i in range(len(newlist)):
            pixel_value = newlist[i]['category_id']
            mask[self.coco.annToMask(newlist[i])==1]=pixel_value

        h0, w0 = mask.shape[:2]  # orig hw
        _r = 640 / max(h0, w0)  # ratio
        if _r != 1:  # if sizes are not equal
            mask = cv2.resize(mask, (int(w0 * _r), int(h0 * _r)), interpolation=cv2.INTER_NEAREST) #% added for segmentation


        return mask
