# Collect Dataset

1. SyncMode/final_collect_dataset.py를 사용해 carla dataset 수집
2. filtering 된 bev target을 만듦 (traffic light, crosswalk 등등 불필요한 부분 수작업으로 제거)
3. SyncMode/saveTarget.py를 사용해 lane, intersection, curb 구별 후 저장

## final_collect_dataset.py

    cd ~/BEVDataset/SyncMode
    python final_collect_dataset.py


## saveTarget.py

    cd ~/BEVDataset/SyncMode
    python saveTarget.py

이 코드는 checkTarget.ipynb에서 확인 할 수 있음.

saveTarget 실행 시 얻을 수 있는 label과 각 RGB value는 다음과 같다.

|Label|Value|
|------|---|
|Ignore|(255, 255, 255)|
|Intersection|(153, 204, 255)|
|Lane|(0, 102, 0)|
|Curb|(255, 0, 0)|

코드 실행시 주의할 점

- 입, 출력 data path 바로 입력하기 / 그렇지 않으면 데이터 날아가기 쉬움