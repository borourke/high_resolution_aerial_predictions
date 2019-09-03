from os import listdir
from PIL import Image
import math
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
from imutils import paths
import numpy as np
import argparse
import os
import cv2

class_selection = input("If you would like to prioritize a type of vehicle please choose (1 - Sedan, 2 - SUV/Truck, 3 - Other, 4 - Unknown, 5 - All): ")

#
# Set up configs
#

OUTPUT_FILE_PATH = "predictions"
CLASSES_FILE = "classes.csv"
MODEL_FILE = "laptop_compiled_model.h5"
GROUND_TRUTH_DIRECTORY = "dataset/ground_truth_images"
TEMPORARY_CROPPED_IMAGE_NAME = 'dataset/temp_cropped_image.jpg'
CONFIDENCE_THRESHOLD = 0.7

# Cropped image size
X_AXIS = 256
Y_AXIS = 256
CROPPED_RATIO = 2
BANDWIDTH = X_AXIS * CROPPED_RATIO

# Ground truth images
ground_truth_images = listdir(GROUND_TRUTH_DIRECTORY)
ground_truth_images.remove(".DS_Store")

# Labels CSV
LABELS = open(CLASSES_FILE).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}
CLASSES = {'1': 'Sedan', '2': 'SUV/Truck', '3': 'Other', '4': 'Unknown'}

# Load the model
model = models.load_model(MODEL_FILE, backbone_name='resnet50')

for ground_truth_image_name in ground_truth_images:
    print(ground_truth_image_name)
    ground_truth_image_path = os.path.sep.join([GROUND_TRUTH_DIRECTORY, ground_truth_image_name])
    ground_truth_image = Image.open(ground_truth_image_path)
    ground_truth_image = ground_truth_image.convert("RGB")
    width, height = ground_truth_image.size
    x_boxes_count = math.ceil(width / X_AXIS) * 2 - 1
    y_boxes_count = math.ceil(height / Y_AXIS) * 2 - 1

    ground_truth_image_predictions = []

    for y_counter in range(y_boxes_count):
        cropped_ymin = y_counter * (Y_AXIS / 2)
        cropped_ymax = cropped_ymin + Y_AXIS
        if(cropped_ymax >= height):
            cropped_ymax = height
            cropped_ymin = cropped_ymax - Y_AXIS
        print("[{}] - Doing y counter for...'{}'".format(ground_truth_image_name, y_counter))
        print("[{}] - Ymin...'{}'".format(ground_truth_image_name, cropped_ymin))
        print("[{}] - Ymax...'{}'".format(ground_truth_image_name, cropped_ymax))
        for x_counter in range(x_boxes_count):
            cropped_xmin = x_counter * (X_AXIS / 2)
            cropped_xmax = cropped_xmin + X_AXIS
            if(cropped_xmax >= width):
                cropped_xmax = width
                cropped_xmin = cropped_xmax - X_AXIS
            print("[{}] -     Doing x counter for...'{}'".format(ground_truth_image_name, x_counter))
            print("[{}] -     Xmin...'{}'".format(ground_truth_image_name, cropped_xmin))
            print("[{}] -     Xmax...'{}'".format(ground_truth_image_name, cropped_xmax))
            cropped = ground_truth_image.crop((cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax))
            wpercent = (BANDWIDTH/float(cropped.size[0]))
            hsize = int((float(cropped.size[1])*float(wpercent)))
            cropped = cropped.resize((BANDWIDTH,hsize), Image.LANCZOS)
            cropped.save(TEMPORARY_CROPPED_IMAGE_NAME, "JPEG")

            # Prediction time
            # load the input image (BGR), clone it, and preprocess it
            image = read_image_bgr(TEMPORARY_CROPPED_IMAGE_NAME)
            image = preprocess_image(image)
            (image, scale) = resize_image(image)
            image = np.expand_dims(image, axis=0)

            # detect objects in the input image and correct for the image scale
            (boxes, scores, labels) = model.predict_on_batch(image)
            boxes /= scale

            # loop over the detections
            for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
                # filter out weak detections
                if(score < CONFIDENCE_THRESHOLD):
                    continue

                if((int(str(LABELS[label])) != int(class_selection)) and (int(class_selection) != 5)):
                    continue

                # convert the bounding box coordinates from floats to integers
                box = box.astype("int")

                # Create the row for each prediction in the format:
                # <classname> <confidence> <ymin> <xmin> <ymax> <xmax>
                row = " ".join([LABELS[label], str(score),
                                str(box[1]), str(box[0]), str(box[3]), str(box[2])])
                # Add each prediction to a set in memory
                # Offset coordinates by image cropping offset
                # Example Filename: <name>-0.png-1.<xmin>-2.<ymin>-3.<xmax>-4.<ymax>-5.txt-6
                xmin = int(box[0]/CROPPED_RATIO)+cropped_xmin
                ymin = int(box[1]/CROPPED_RATIO)+cropped_ymin
                xmax = int(box[2]/CROPPED_RATIO)+cropped_xmin
                ymax = int(box[3]/CROPPED_RATIO)+cropped_ymin

                ground_truth_image_predictions.append({'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax), 'ymax': int(ymax), 'class': LABELS[label], 'confidence': score})

    print(ground_truth_image_predictions)

    img = cv2.imread(ground_truth_image_path)
    for prediction in ground_truth_image_predictions:
        object_class = CLASSES[str(prediction['class'])]
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (prediction['xmax'],prediction['ymax'])
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        top_left = (int(prediction['xmin']), int(prediction['ymax']))
        bottom_left = (int(prediction['xmin']), int(prediction['ymin']))
        bottom_right = (int(prediction['xmax']), int(prediction['ymin']))
        top_right = (int(prediction['xmax']), int(prediction['ymax']))
        x_length = int(prediction['xmax']) - int(prediction['xmin'])
        y_length = int(prediction['ymax']) - int(prediction['ymin'])
        centroid = (int(int(prediction['xmin']) + (x_length/2)), int(int(prediction['ymin'] + (y_length/2))))
        # rectangle
        if(object_class == 'SUV/Truck'):
            img = cv2.rectangle(img, top_left, bottom_right, (255,255,255), 1)
            print("DOING RECTANGLE")
        # triangle
        elif(object_class == 'Sedan'):
            pt1 = (int(prediction['xmin']) + (x_length/2), int(prediction['ymax']))
            pt2 = bottom_left
            pt3 = bottom_right
            pts = np.array([pt1, pt2, pt3], dtype=np.int32)
            img = cv2.polylines(img, [pts], isClosed=True, color=(255,255,255), thickness=1)
            print("DOING TRIANGLE")
        # circle
        elif(object_class == 'Other'):
            img = cv2.circle(img, centroid, int(x_length/2), (255,255,255))
            print("DOING CIRLCE")
        else:
            img = cv2.circle(img, centroid, int(x_length/2), (255,255,255))
            print("DOING CIRCLE")

    cv2.imwrite(os.path.sep.join([OUTPUT_FILE_PATH, "predicted-{}".format(ground_truth_image_name)]), img)

    ground_truth_image = Image.open(ground_truth_image_path)
    counter = 0
    basewidth = 128
    for prediction in ground_truth_image_predictions:
        if(counter >= 5):
            break
        counter += 1
        cropped = ground_truth_image.crop((int(prediction['xmin']), int(prediction['ymin']), int(prediction['xmax']), int(prediction['ymax'])))
        wpercent = (basewidth/float(cropped.size[0]))
        hsize = int((float(cropped.size[1])*float(wpercent)))
        cropped = cropped.resize((basewidth,hsize), Image.ANTIALIAS)
        cropped.show()