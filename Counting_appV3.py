# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:44:14 2024

@author: mohan
"""

#importing required packages
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from supervision.detection.core import Detections
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from supervision.detection.polygon_zone import PolygonZone, PolygonZoneAnnotator
from strong_sort.strong_sort import StrongSORT
import os
import torch

# List of Coordinates
temp_coordinates = []
coordinates_list = []

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
strong_sort_wts = "resnet50_msmt17.pt"
tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = False, ema_alpha=0.8, max_age = 1000, max_dist = 0.15, max_iou_distance = 0.999, mc_lambda = 0.9999)
# tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = True, 
#                       ema_alpha=0.89, max_age = 10000, max_dist = 0.16, max_iou_distance = 0.543, mc_lambda = 0.995)


# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = (x, y)
        #print(f"Clicked at: {coordinates}")
        temp_coordinates.append(coordinates)
    
        
# Line Counting Function       
def Line_Counter(coordinates_list):
    point_list = []
    for i,j in coordinates_list:
        point_list.append(sv.Point(i, j))
    if len(point_list) == 2:
        return LineZone(start=point_list[0], end=point_list[1]) #line_counter
    else:
        return None
    
# Area Counting Function
def Area_counter(coordinates_list, frame_wh):
    zones = []
    for area in coordinates_list:
        if len(area) > 2:
            zones.append(PolygonZone(polygon=area, frame_resolution_wh=frame_wh))
    return zones

# Open camera
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

model = YOLO('yolov8s.pt')
class_names = model.names
class_id_mapping = {idx: name for idx, name in enumerate(class_names)}

line_counter = None
zones = []
zone_annotators = []

box_annotator = sv.BoxAnnotator(thickness=2,
                                text_thickness=2,
                                text_scale=1)

line_annotator = LineZoneAnnotator(thickness=2,
                                   text_thickness=2,
                                   text_scale=1)

cv2.namedWindow('Object Counter')

detections = []

color = sv.ColorPalette.default()

while True:
    success, frame = camera.read()

    if not success:
        print("Camera not found!")
        break
    results = model(source=frame, classes=0) # person, TV, laptop, cellphone, keyboard
    cv2.setMouseCallback("Object Counter", click_event)
    for coordinates in temp_coordinates:
        frame = cv2.circle(frame, coordinates, radius=3, color=(0, 0, 255), thickness=-1)
    
    if cv2.waitKey(1) & 0xFF == ord('l'):
        print('Line Counter Function initiated')
        line_counter = Line_Counter(temp_coordinates)
        temp_coordinates = []  # Clear after setting the line counter
    
    if cv2.waitKey(1) & 0xFF == ord('a'):
        print('Area Counter Function initiated')
        frame_wh = (frame.shape[1], frame.shape[0])  # Get frame width and height
        if len(temp_coordinates) > 2:
            coordinates_list.append(np.array(temp_coordinates))
            temp_coordinates = []  # Clear for the next zone
        
        zones = Area_counter(coordinates_list, frame_wh)
        zone_annotators = [PolygonZoneAnnotator(zone=zone, color=color.by_idx(index+1), thickness=2, text_thickness=2, text_scale=1) 
                           for index, zone in enumerate(zones)]
    
    bboxes_xywh = []
    confs = []
    classes = []
    
    #detections = sv.Detections.from_yolov8(result)
    for result in results:
        boxes1 = result.boxes.xywh.tolist()
        class_ids = result.boxes.cls
        confidences = np.array(result.boxes.conf.tolist())
        class_labels = [class_id_mapping[int(class_id)] for class_id in class_ids]     
        bboxes_xywh.extend(boxes1)
        confs.extend(confidences)
        classes.extend(class_labels)
    
    bboxes_xywh = np.array(bboxes_xywh)
    confs = torch.tensor(confs)  # Convert confs to PyTorch tensor
    classes = torch.tensor(classes) 
    
    tracks = tracker.update(bboxes_xywh, confs, classes, frame)

    track_confs = [track.conf for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
    track_classes = [track.class_id for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
    track_ids = [track.track_id for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
    track_dets = [track.to_tlbr() for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
    
    
    
    #print(track_confs)
    
    if len(track_dets) > 0:
        
        detections = Detections(
                        xyxy = np.array(track_dets),
                        confidence = np.array(track_confs),
                        class_id   = np.array(track_classes).astype(int),
                        tracker_id = np.array(track_ids)
                    )

    # detections = sv.Detections.from_yolov8()    
    
    #mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    #detections.filter(mask=mask, inplace=True)

        labels = [f'{model.model.names[class_id]} ID: {track_ID}|{confidence:0.2f}' for _, confidence, class_id, track_ID in detections]
        frame =box_annotator.annotate(
            scene = frame,
            detections = detections,
            labels = labels
            )
    
    if line_counter:
        line_annotator.annotate(frame, line_counter)
        if len(detections) > 0:
            line_counter.trigger(detections=detections)
    
    if zones:
        for zone, zone_annotator in zip(zones, zone_annotators):
            zone_annotator.annotate(frame)
            if len(detections) > 0:
                zone.trigger(detections=detections)
        
    cv2.imshow('Object Counter', frame)    
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break
    elif cv2.waitKey(1) & 0xFF == 8:
        if len(coordinates_list) > 0:
            coordinates_list.pop()
            print("Removed previous points")
        else:
            print("All points are removed")
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        zones = []
        line_counter = None
        coordinates_list = []
        temp_coordinates = []
        zone_annotators = []
        
camera.release()
cv2.destroyAllWindows()