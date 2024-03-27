import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pyttsx3
import time

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
model=YOLO('yolov8n.pt')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count=0

# Initialize variables for object tracking
prev_centroids = {}  # Dictionary to store centroids of objects in the previous frame

# Camera parameters (to be adjusted based on your camera and setup)
KNOWN_WIDTH = 18  # Width of the object in cm (adjust according to your object)
FOCAL_LENGTH = 640  # Focal length of the camera in pixels (adjust according to your camera)

def calculate_distance(known_width, focal_length, object_width):
    return (known_width * focal_length) / object_width / 100  # Divide by 100 to convert cm to meters

def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))  # Compute Euclidean distance between two points

while True:
    im= picam2.capture_array()
    
    count += 1
    if count % 3 != 0:
        continue
    im=cv2.flip(im,-1)
    results=model.predict(im)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    
    curr_centroids = {}  # Dictionary to store centroids of objects in the current frame

    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
        cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)

        # Calculate centroid of the bounding box
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        curr_centroids[c] = centroid  # Store centroid in current centroids dictionary

        # Check if object was detected in previous frame
        if c in prev_centroids:
            # Calculate Euclidean distance between centroids
            distance_px = euclidean_distance(prev_centroids[c], centroid)

            # Calculate the width of the detected object
            object_width = x2 - x1

            # Convert distance from pixels to meters
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, object_width)

            # Voice output for object and distance
            voice_feedback = f"{c} moved. Distance is {distance:.2f} meters."
            engine.say(voice_feedback)
            engine.runAndWait()

    # Update previous centroids with current centroids for next iteration
    prev_centroids = curr_centroids

    cv2.imshow("Camera", im)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
