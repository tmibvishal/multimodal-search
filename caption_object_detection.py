# Helped taken from tutorial provided in the assignment statement
# https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
import json
import os
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2

from config import min_confidence


def get_coco_object_categories():
    coco_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                   "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                   "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                   "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate",
                   "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                   "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door",
                   "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender",
                   "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]
    coco_labels_dict = {}
    for i, label in enumerate(coco_labels):
        coco_labels_dict[i + 1] = label
    return coco_labels_dict


def torch_preprocess(orig_image):
    image = orig_image.copy()

    # convert the image from BGR to RGB channel ordering and change the
    # image from channels last to channels first ordering
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    return image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = get_coco_object_categories()
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize a dictionary containing model name and its corresponding
# torchvision function call
MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn
}

# load the model and set it to evaluation mode
model = MODELS['frcnn-resnet'](pretrained=True, progress=True,
                                   num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

def get_captions_object_detection(img_path):
    # set the device we will be using to run the model
    

    words = []

    orig_image = cv2.imread(img_path)
    image = torch_preprocess(orig_image)

    # send the input to the device and pass the it through the network to
    # get the detections and predictions
    image = image.to(DEVICE)
    detections = model(image)[0]

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections["scores"][i].item()
        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > min_confidence:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")

            words.append(CLASSES[idx])

            # # display the prediction to our terminal
            # label = "{}: {:.2f}%".format('person', confidence * 100)
            # print("[INFO] {}".format(label))
            # # draw the bounding box and label on the image
            # cv2.rectangle(orig_image, (startX, startY), (endX, endY), COLORS[idx], 2)
            # y = startY - 15 if startY - 15 > 15 else startY + 15
            # cv2.putText(orig_image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output image
    # cv2.imshow("Output", orig_image)
    # cv2.waitKey(10)
    return words
