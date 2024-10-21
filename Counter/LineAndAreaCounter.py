# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:36:53 2024

@author: mohan
"""

import cv2
from ultralytics import YOLO
import supervision as sv
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from supervision.detection.core import Detections
import numpy as np
#from strong_sort.strong_sort import StrongSORT
import torch
import os

def Line_Counter(coordinates_list, frame):
    # LINE_START = sv.Point(320, 470)
    # LINE_END = sv.Point(320, 10)
    point_list = []
    for i,j in coordinates_list:
        point_list.append(sv.Point(i, j))
    
    line_counter = LineZone(start=point_list[0], end=point_list[1])

    line_annotator = LineZoneAnnotator(thickness=2,
                                       text_thickness=2,
                                       text_scale=1)

    #line_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
    box_annotator = sv.BoxAnnotator(thickness =2,
                                    text_thickness = 2,
                                    text_scale = 1)
