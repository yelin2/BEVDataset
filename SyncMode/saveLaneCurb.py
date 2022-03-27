
import cv2
import numpy as np
from glob import glob

# read & refine
filenames = glob('/Users/yelin/Downloads/target_long/bev/6030m640320p_filtered/*.png')


for filename in filenames:
    bev = cv2.imread(filename)
    bev = np.mean(bev, 2)

    bev[bev<42] = 0
    road = (bev>43)*(bev<=127)
    bev[road] = 85

    inter = (bev>127)*(bev<=212)
    bev[inter] = 170
    
    bev[bev>212] = 255
    print(np.unique(bev))


    # seperate lane, road, intersection
    lane = np.zeros_like(bev, dtype=np.uint8)
    lane[bev==255] = 255

    road = np.zeros_like(bev, dtype=np.uint8)
    road_mask = np.logical_or(bev==255, bev==85)
    road_mask = np.logical_or(road_mask, bev==170)
    road[road_mask] = 255

    inter = np.zeros_like(bev, dtype=np.uint8)
    inter[bev==170] = 255


    # calculate curb using canny edge detection
    edges = cv2.Canny(road, 100, 200)
    print(edges.shape)
    ex_idx, ey_idx = np.where(edges == 255)

    print(ey_idx.shape)
    min_ex_idx = ex_idx-3
    min_ex_idx[min_ex_idx < 0] = 0

    max_ex_idx = ex_idx+3
    max_ex_idx[max_ex_idx > 512] = 512

    min_ey_idx = ey_idx-3
    min_ey_idx[min_ey_idx < 0] = 0

    max_ey_idx = ey_idx+3
    max_ey_idx[max_ey_idx > 512] = 512

    print(max_ey_idx.shape)
    for min_x, max_x, min_y, max_y in zip(min_ex_idx, max_ex_idx, min_ey_idx, max_ey_idx):
        edges[min_x:max_x, min_y:max_y] = 255


    # save
    filename = 'target/'
    cv2.imwrite(filename + 'Lane.png', lane.astype(np.uint8))
    cv2.imwrite(filename + 'Road.png', road.astype(np.uint8))
    cv2.imwrite(filename + 'Intersection.png', inter.astype(np.uint8))
    cv2.imwrite(filename + 'Curb.png', edges.astype(np.uint8))


# for filename in filenames:
#     bev = cv2.imread(filename)
#     print(bev.shape)
