# Generate Target

Generate BEV target from CARLA
1. semantic segmentation for road, roadline
2. object detection for dynamic objects
3. instance segmentation for dynamic objects


## 1. Run CARLA

    cd ~/opt/carla-simulator
    ./CarlaUE4.sh
## 2. Visualize Semantic segmnetation

class: road, roadlines

    # new terminal
    cd ~/ws/BEVDataset/generateTarget
    
    # for road segmentation
    python visSegRoad.py

    # for vehicle segmentation
    python visSegVehicle.py


## 3. Visualize Object Detection & Instance Segmentation

class: pedestrians, vehicles

    # new terminal
    cd ~/ws/BEVDataset/generateTarget
    python visDet.py

defualt = not include occluded bbox 

problem = occluded bbox will be target

if you want to include occluded bbox, uncommnet visDet.py 349, 356 line


## 4. Save dynamic objects
this will be save below target

1. object detection (vehicle, pedestrian)
2. instance segmentation (vehicle, pedestrian)
3. semantic segmentation (vehicle)

save path root = '~/BEVData/'

        cd ~/ws/BEVDataset/generateTarget
        python saveSegRoad.py
## 5. Save static objects
this will be save below target

1. semantic segmentation (road, roadline, traffic light)

a

        cd ~/ws/BEVDataset/generateTarget
        python saveSegRoad.py
