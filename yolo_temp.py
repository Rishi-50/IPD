import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pyttsx3
import time
import math
import threading

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
model = YOLO('yolov8n.pt')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0

# Camera parameters (to be adjusted based on your camera and setup)
KNOWN_WIDTH = 18  # Width of the object in cm (adjust according to your object)
FOCAL_LENGTH = 640  # Focal length of the camera in pixels (adjust according to your camera)

def calculate_distance(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width / 100  # Divide by 100 to convert cm to meters

def process_frame(im):
    global count
    count += 1
    if count % 3 != 0:
        return

    im = cv2.flip(im, -1)
    results = model.predict(im)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cvzone.putTextRect(im, f'{c}', (x1, y1), 1, 1)

        # Calculate distance
        box_width = x2 - x1
        distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)

        # Voice output for object and distance
        voice_feedback = f"{c} detected. Distance is {distance:.2f} meters."
        engine.say(voice_feedback)
        engine.runAndWait()

        # Pause for 1 second
        time.sleep(1)

    cv2.imshow("Camera", im)

def capture_frames():
    while True:
        im = picam2.capture_array()
        process_frame(im)

def main():
    threading.Thread(target=capture_frames, daemon=True).start()
    while True:
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
