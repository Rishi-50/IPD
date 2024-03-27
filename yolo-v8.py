import cv2
import numpy as np
import pyttsx3
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
from ultralytics import YOLO

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

# Warm-up camera
time.sleep(2)

# Load YOLO model
model = YOLO('yolov5s.pt')

# Object classes
classNames = model.module.names

# Camera parameters (to be adjusted based on your camera and setup)
KNOWN_WIDTH = 18  # Width of the object in cm (adjust according to your object)
FOCAL_LENGTH = 640  # Focal length of the camera in pixels (adjust according to your camera)

def calculate_distance(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width / 100  # Divide by 100 to convert cm to meters

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    results = model(image)

    for det in results.xyxy[0]:
        x1, y1, x2, y2, _, class_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_name = classNames[int(class_id)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
    cv2.imshow("Camera", image)
    
    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    # Exit condition
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
camera.close()
