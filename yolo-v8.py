import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import pyttsx3
import time
import math

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

# Initialize PiCamera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLO model
model = YOLO('yolov8n.pt')

# Load class list
with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")

# Camera parameters
KNOWN_WIDTH = 18  # Width of the object in cm (adjust according to your object)
FOCAL_LENGTH = 640  # Focal length of the camera in pixels (adjust according to your camera)

# Function to calculate distance
def calculate_distance(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width / 100  # Convert cm to meters

# Main loop for object detection
while True:
    im = picam2.capture_array()
    results = model.predict(im)

    for det in results.pred[0]:
        x1, y1, x2, y2, _, class_id, _ = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_name = class_list[int(class_id)]
        
        # Draw bounding box
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(im, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate distance
        box_width = x2 - x1
        distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)
        
        # Voice feedback
        voice_feedback = f"{class_name} detected. Distance is {distance:.2f} meters."
        engine.say(voice_feedback)
        engine.runAndWait()
        
        # Pause for 1 second
        time.sleep(1)

    # Display the image
    cv2.imshow("Camera", im)
    
    # Exit condition
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
