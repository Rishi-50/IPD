import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import pandas as pd
import onnxruntime
import cvzone
import numpy as np
import pyttsx3
import time

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

# Initialize the PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
raw_capture = PiRGBArray(camera, size=(640, 480))

# Load the ONNX model
model = onnxruntime.InferenceSession('yolov8n.onnx')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Camera parameters (to be adjusted based on your camera and setup)
KNOWN_WIDTH = 2  # Width of the object in cm (adjust according to your object)
FOCAL_LENGTH = 640  # Focal length of the camera in pixels (adjust according to your camera)

# Initialize variables for object tracking
prev_centroids = {}  # Dictionary to store centroids of objects in the previous frame

# Function to calculate distance
def calculate_distance(known_width, focal_length, object_width):
    return (known_width * focal_length) / object_width   # Divide by 100 to convert cm to meters

# Function to calculate Euclidean distance
def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

# Start capturing video from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # Grab the raw NumPy array representing the image
    im = frame.array

    # Perform object detection
    results = model.predict(im)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    curr_centroids = {}  # Dictionary to store centroids of objects in the current frame

    for index, row in px.iterrows():
        x1, y1, x2, y2, d = map(int, row[:5])
        c = class_list[d]

        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cvzone.putTextRect(im, f'{c}', (x1, y1), 1, 1)

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
            voice_feedback = f"{c} detected. Distance is {distance:.2f} centimeters."
            engine.say(voice_feedback)
            engine.runAndWait()

            time.sleep(2)

    # Update previous centroids with current centroids for next iteration
    prev_centroids = curr_centroids

    # Display the frame
    cv2.imshow("Camera", im)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Clear the stream for the next frame
    raw_capture.truncate(0)

# Release the camera and close all OpenCV windows
camera.close()
cv2.destroyAllWindows()
