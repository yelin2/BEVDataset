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
    python visSeg.py