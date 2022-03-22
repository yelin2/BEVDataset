
import cv2
import numpy as np
from glob import glob
import torch
from torch import nn

# read & refine
filenames = glob('target/bev/6030m512256p/*.png')
filenames = ['target/bev/000001_filtered.png']
pool160x40 = nn.AdaptiveMaxPool2d((160,80))


for i, filename in enumerate(filenames):
    bev = cv2.imread(filename)
    bev = np.mean(bev, 2)

    bev[bev<42] = 0
    road = (bev>43)*(bev<=127)
    bev[road] = 85

    inter = (bev>127)*(bev<=212)
    bev[inter] = 170
    
    bev[bev>212] = 255
    print(np.unique(bev))


    bev_160x40 = pool160x40(torch.from_numpy(bev[None, None, :, :])).squeeze(0).squeeze(0).numpy().astype(np.uint8)


    # save
    filename = 'target/bev/000001_filtered_resized.png'
    # filename = 'target/bev/6030m512256p_resized/'

    cv2.imwrite(filename + f'{i}.png', bev_160x40)



# for filename in filenames:
#     bev = cv2.imread(filename)
#     print(bev.shape)
