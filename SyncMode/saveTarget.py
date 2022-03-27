
'''
    by Yelin,

    lane, intersection이 labeling 된 image로 부터 lane, curb, intersection 분리

    Input Seg Image (Gray Image)
        Lable        | Value
        -----------------------
        Ignore       | 0
        Road         | 85
        Intersection | 170
        Lane         | 255

    Output Seg Image (Gray Image)
        Lable        | Value
        -----------------------
        Ignore       | (255, 255, 255)
        Intersection | (153, 204, 255)
        Lane         | (0, 102, 0)
        Curb         | (255, 0, 0)

    저장은 다 따로 하는거(for training) + 하나로 합쳐서 하는거(for visualization)
    따로 하는 이유는 직선 차선에서 lane, curb가 겹치는 점이 많기 때문 -> loss를 BCE로 주기 때문에 겹쳐도 상관 없음
    
'''


import cv2
import numpy as np
from glob import glob


# read & refine
filenames = glob('/Users/yelin/Downloads/target_long/bev/6030m640320p_filtered/*.png')


for i, filename in enumerate(filenames):

    if i % 50 == 0:
        print(i)

    # read target bev
    bev = cv2.imread(filename)


    # distinguish road(85), lane(255), intersection(170), ignore(0)
    bev = np.mean(bev, 2)

    bev[bev<42] = 0
    road = (bev>43)*(bev<=127)
    bev[road] = 85

    inter = (bev>127)*(bev<=212)
    bev[inter] = 170
    
    bev[bev>212] = 255


    # generate lane, intersection target
    h, w = bev.shape

    lane_tgt = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
    intr_tgt = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

    lane_tgt[bev==255,:] = (0, 102, 0)
    intr_tgt[bev==170, :] = (153, 204, 255)


    # generate target for visualization
    all_tgt = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

    all_tgt[bev==255,:] = (0, 102, 0)
    all_tgt[bev==170, :] = (153, 204, 255)


    # generate curb target
    curb_tgt = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

    road = np.zeros_like(bev, dtype=np.uint8)
    road_mask = np.logical_or(bev==255, bev==85)
    road_mask = np.logical_or(road_mask, bev==170)
    road[road_mask] = 255

    # calculate curb using canny edge detection
    edges = cv2.Canny(road, 100, 200)

    curb_tgt[edges==1, :] =(255, 0, 0)

    # target.shape
    ex_idx, ey_idx = np.where(edges == 255)

    min_ex_idx = ex_idx-1
    min_ex_idx[min_ex_idx < 0] = 0

    max_ex_idx = ex_idx+1
    max_ex_idx[max_ex_idx > 512] = 512

    min_ey_idx = ey_idx-1
    min_ey_idx[min_ey_idx < 0] = 0

    max_ey_idx = ey_idx+1
    max_ey_idx[max_ey_idx > 512] = 512

    for min_x, max_x, min_y, max_y in zip(min_ex_idx, max_ex_idx, min_ey_idx, max_ey_idx):
        curb_tgt[min_x:max_x, min_y:max_y, :] = (255, 0, 0)
        all_tgt[min_x:max_x, min_y:max_y, :] = (255, 0, 0)


    # save
    fileroot = '/Users/yelin/Downloads/target_long/bev/6030m640320p_final/'
    filename = filename.split('/')[-1]

    # cv2 format is BGR, so convert RGB to BGR befor save
    cv2.imwrite(fileroot + 'vis/' + filename, all_tgt.astype(np.uint8)[:,:,::-1])
    cv2.imwrite(fileroot + 'lane/' + filename, lane_tgt.astype(np.uint8)[:,:,::-1])
    cv2.imwrite(fileroot + 'intersection/' + filename, intr_tgt.astype(np.uint8)[:,:,::-1])
    cv2.imwrite(fileroot + 'curb/' + filename, curb_tgt.astype(np.uint8)[:,:,::-1])

