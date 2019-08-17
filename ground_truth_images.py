import os
from os import listdir
import cv2
from PIL import Image
import math

X_AXIS = 256
Y_AXIS = 256

grouth_truth_directory = 'dataset/ground_truth_images/'

ground_truth_image_name = 'top_potsdam_2_10_RGB.png'
ground_truth_image_path = os.path.sep.join([grouth_truth_directory, ground_truth_image_name])
image = Image.open(ground_truth_image_path)
width, height = image.size
x_boxes_count = math.ceil(width / X_AXIS)
y_boxes_count = math.ceil(height / Y_AXIS)

for y_counter in range(y_boxes_count):
    ymin = y_counter * Y_AXIS
    ymax = ymin + Y_AXIS
    if(ymax >= height):
        ymax = height
        ymin = ymax - Y_AXIS
    print("Doing y counter for...'{}'".format(y_counter))
    print("Ymin...'{}'".format(ymin))
    print("Ymax...'{}'".format(ymax))
    for x_counter in range(x_boxes_count):
        xmin = x_counter * X_AXIS
        xmax = xmin + X_AXIS
        if(xmax >= width):
            xmax = width
            xmin = xmax - X_AXIS
        print("    Doing x counter for...'{}'".format(x_counter))
        print("    Xmin...'{}'".format(xmin))
        print("    Xmax...'{}'".format(xmax))
        cropped = image.crop((xmin,ymin,xmax,ymax))
        cropped.save("dataset/cropped_images/{}.{}.{}.{}.{}.jpg".format(ground_truth_image_name, xmin, ymin, xmax, ymax), "JPEG")

predict_command = "python /Users/bryanorourke/Desktop/iPadU2/aeriel-models/bryan_retinanet/predict.py --model /Users/bryanorourke/Desktop/iPadU2/aeriel-models/bryan_retinanet/laptop_compiled_model.h5 --input dataset/cropped_images --confidence 0.3"

os.system(predict_command)
