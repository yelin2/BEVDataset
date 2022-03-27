# BEVDataset

## Documentation TODO
1. args 어떤 것들 있는지 표로 정리
2. tick sensor, world.tick() 동작 원리 정리
3. sensor manager - callback 함수 동작 원리 정리
4. save target 어떻게 하는지 정리
5. utils dir에 있는 파일들 cva import path 정리


## Code TODO
1. ~~hz 100으로 설정 되어 있는데 10 hz로 해도 되는거 아닌가~~ -> 10hz로 변경
2. ~~CarlaSyncMode class 안쓰는 이유~~
    -> 이미 code에 다 명시되어 있음
3. ~~args w, h 받아서 *3 하는 이유~~
    -> display manager 에서 window size 정해줄라고 했는데 지금 window size fix 되어 있어서 지워도 됨
4. layered map 추가 -> segmentation class랑 같이 다시 생각 해 볼 것

    parkedvehicles, props, streetlights, decals, foliage


    RGB: 

    dynamic target

    static target

5. --show True option 주면 save 제대로 안 됨
    민주 -> 빼먹은 코드 있나 확인 해 주기

6. carla map 어디서 돌릴지 정하기
    혜민 -> 각 map의 크기 조사 & 띄워서 보기

7. collect 한 data visualize 코드 필요
