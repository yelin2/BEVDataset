import numpy as np
import math

def whfov_to_intrinsic(ImageSizeX, ImageSizeY, CameraFOV):

    ImageSizeX = 964
    ImageSizeY = 604
    CameraFOV = 70.4

    fx = ImageSizeX /(2 * np.tan(CameraFOV * np.pi / 360))
    fy = ImageSizeY /(2 * np.tan(CameraFOV * np.pi / 360))
    Center_X = ImageSizeX / 2
    Center_Y = ImageSizeY / 2

    print(fx,", ", fy)
    print (Center_X,", ", Center_Y)

def whfocal_to_fov(w, h, fx, fy):

    rad_fov_x = 2 * math.atan2(w,(2*fx))
    rad_fov_y = 2 * math.atan2(h,(2*fy))

    deg_fov_x = np.degrees(rad_fov_x)
    deg_fov_y = np.degrees(rad_fov_y)

    print(deg_fov_x, ", ", deg_fov_y)

def bevrange_meter(x_deg, y_deg, camheight):
    #x_bev_deg /2, y_bev_deg/2, camheight

    x_rad = np.radians(x_deg)
    y_rad = np.radians(y_deg)

    x_c = camheight * math.tan(x_rad)
    y_c = camheight * math.tan(y_rad)

    print("bevrange_meter", x_c*2, ", ", y_c*2)

def meter_per_pixel(w, h, x_meter, y_meter):
    print("meter_per_pixel: ", x_meter/w,", ", y_meter/h)




def metersetting(w, h, x_meter, y_meter, fov):
    fx = w /(2 * np.tan(fov * np.pi / 360))
    fy = h /(2 * np.tan(fov * np.pi / 360))
    cx = w/2
    cy = h/2

    x_rad = np.radians(fov/2)
    print("metersetting meter per pixel: ", x_meter/w, y_meter/h)
    cam_height = x_meter/np.tan(x_rad)

    return w, h, cam_height/2


if __name__ == '__main__':
    w = 480
    h = 480
    x_meter = 100
    y_meter = 100
    fov = 70.4
    camheight = 35.44

    print("meter setting: ",metersetting(w, h, x_meter, y_meter, fov))
    # print("------------------")
    # bevrange_meter(fov/2,fov/2 , camheight)
    # meter_per_pixel(w, h, x_meter, y_meter)



