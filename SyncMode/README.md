# Collect Dataset

## 1. save dynamic objects

    cd ~/BEVDataset/SyncMode
    python saveDynamics.py --data_root {your data root} --scene_num {drive scene number}

    # example
    python saveDynamics.py --data_root /home/rml/ws/TestData --scene_num /drive_001

## 2. save static objects

    cd ~/BEVDataset/SyncMode
    python saveStatics.py --data_root {your data root} --scene_num {drive scene number} --is_static True

    # example
    python saveStatics.py --data_root /home/rml/ws/TestData --scene_num /drive_001 --is_static True

    