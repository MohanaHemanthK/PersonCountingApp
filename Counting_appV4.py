# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:04:46 2024

@author: mohan
"""

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
import pyzed.sl as sl
from sys import exit

# List of Coordinates
temp_coordinates = []
coordinates_list = []

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
# camera = cv2.VideoCapture(1)

# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

zed = sl.Camera()

init_params = sl.InitParameters()

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
    print("Camera Open : "+repr(status)+". Exit program.")
    exit()

# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()

image = sl.Mat()

model = YOLO('yolov8s.pt')
class_names = model.names

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
    #success, frame = camera.read()

    # if not success:
    #     print("Camera not found!")
    #     break
     if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
         
         # Retrieve left image
         zed.retrieve_image(image, sl.VIEW.LEFT)
         # Retrieve depth map. Depth is aligned on the left image

         frame = image.get_data()
         
         
         frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
         results = model.track(source= frame, classes=0) # person, TV, laptop, cellphone, keyboard
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
        
         for result in results:
             boxes = result.boxes.xyxy.tolist()
             class_ids = result.boxes.cls
             confidences = np.array(result.boxes.conf.tolist())
             class_labels = [int(class_id) for class_id in class_ids]
             if result.boxes.is_track:
                 ids = result.boxes.id.tolist()
                 detections = Detections(
                                xyxy=np.array(boxes),
                                confidence=np.array(confidences),
                                class_id=np.array(class_labels).astype(int),
                                tracker_id=np.array(ids)
                          )
                
                 labels = [f'{model.model.names[class_id]} ID: {track_ID}|{confidence:0.2f}' for _, confidence, class_id, track_ID in detections]
                 frame = box_annotator.annotate(
                    scene=frame,
                    detections=detections,
                    labels=labels
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
        
zed.close()
cv2.destroyAllWindows()
