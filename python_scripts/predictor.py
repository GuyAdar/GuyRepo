#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Predictor:
    def __init__(self, args):
        config_path  = args['conf']
        weights_path = args['weights']

        with open(config_path) as config_buffer:
            self.config = json.load(config_buffer)

        self.yolo = YOLO(backend        = self.config['model']['backend'],
                    input_size          = self.config['model']['input_size'],
                    labels              = self.config['model']['labels'],
                    max_box_per_image   = self.config['model']['max_box_per_image'],
                    anchors             = self.config['model']['anchors'])

        self.yolo.load_weights(weights_path)

    def predict(self, image_path):
        image = cv2.imread(image_path)
        boxes = self.yolo.predict(image)
        labels = self.config['model']['labels']
        image_h, image_w, _ = image.shape

        annots = []
        for box in boxes:
            annots.append([
                box.xmin*image_w,
                box.ymin*image_h,
                box.xmax*image_w,
                box.ymax*image_h,
                int(labels[box.label])
            ])
        return annots

        # image = draw_boxes(image, boxes, self.config['model']['labels'])

        # print(len(boxes), 'boxes are found')

        # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
