import numpy as np
import glob

files = glob.glob('./vehicle_bbox/*.txt')
# files = glob.glob('./walker_bbox/*.txt')

image_num = len(files)

object_num_b = 0

x_min_b = 1000
y_min_b = 1000

x_max_b = 0
y_max_b = 0

sum_x_b = 0
sum_y_b = 0

object_num_r = 0

x_min_r = 1000
y_min_r = 1000

x_max_r = 0
y_max_r = 0

sum_x_r = 0
sum_y_r = 0

for i in range(len(files)):
    f = open(files[i], 'r', encoding='utf-8')
    data = f.read()
    f.close()
    data = eval(data)

    object_num_b += len(data['bboxes'])
    for i in range(len(data['bboxes'])):
        x = abs(abs(data['bboxes'][i][1][0]) - abs(data['bboxes'][i][0][0]))
        y = abs(abs(data['bboxes'][i][1][1]) - abs(data['bboxes'][i][0][1]))
        
        sum_x_b += x
        sum_y_b += y

        if x < x_min_b:
            x_min_b = x

        if y < y_min_b:
            y_min_b = y

        if x > x_max_b:
            x_max_b = x

        if y > y_max_b:
            y_max_b = y
    

    object_num_r += len(data['removed_bboxes'])
    for i in range(len(data['removed_bboxes'])):
        x = abs(abs(data['removed_bboxes'][i][1][0]) - abs(data['removed_bboxes'][i][0][0]))
        y = abs(abs(data['removed_bboxes'][i][1][1]) - abs(data['removed_bboxes'][i][0][1]))
        
        sum_x_r += x
        sum_y_r += y

        if x < x_min_r:
            x_min_r = x

        if y < y_min_r:
            y_min_r = y

        if x > x_max_r:
            x_max_r = x

        if y > y_max_r:
            y_max_r = y

print("number of images:",image_num)
print("-------------min max-------------")


print("bboxes min x, min y:", x_min_b,",", y_min_b)
print("bboxes max x, max y:", x_max_b, ",", y_max_b)

print("removed min x, min y:", x_min_r,",",  y_min_r)
print("removed max x, max y:", x_max_r,",",  y_max_r)

print("\n-------------only bboxes-------------")
print("average",object_num_b, "objects x,y:",sum_x_b/object_num_b,",", sum_y_b/object_num_b)

print("\n-------------including removed-------------")
print("average",object_num_r + object_num_b, "objects x,y:",(sum_x_b+ sum_x_r)/(object_num_r+object_num_b) ,",", (sum_y_b+ sum_y_r)/(object_num_r+ object_num_b),"\n" )
