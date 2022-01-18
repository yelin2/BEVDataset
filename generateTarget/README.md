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

1. RGB image
2. object detection (vehicle, pedestrian)
3. instance segmentation (vehicle, pedestrian)
4. semantic segmentation (vehicle)

### Data path
RGB image: TestDataset/image_00

obj det, instance segmentation: TestDataset/object_detection

semantic segmentation: TestDataset/Segmentation/dynamic

### Execution
        cd ~/ws/BEVDataset/generateTarget
        python saveDynamics.py 
                    --data_root "/home/rml/ws/TestDataset/"
                    --scene_num "drive_001"

## 5. Save static objects
this will be save below target

1. RGB image
2. semantic segmentation (road, roadline, traffic light)

### Data path
RGB image: TestDataset/image_01

semantic segmentation: TestDataset/Segmentation/static

### Execution
data_root: dataset root. if directory doesn't exist, code will make specified directories
scene_num: driving scene number (ex. drive_001)

        cd ~/ws/BEVDataset/generateTarget
        python saveStatics.py 
                    --data_root "/home/rml/ws/TestDataset/"
                    --scene_num "drive_001"
