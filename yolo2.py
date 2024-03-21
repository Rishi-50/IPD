import cv2
import math
import pyttsx3
from ultralytics import YOLO

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Camera parameters (to be adjusted based on your camera and setup)
KNOWN_WIDTH = 18  # Width of the object in cm (adjust according to your object)
FOCAL_LENGTH = 640  # Focal length of the camera in pixels (adjust according to your camera)

def calculate_distance(known_width, focal_length, percieved_width):
    return (known_width * focal_length) / percieved_width / 100  # Divide by 100 to convert cm to meters

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Calculate distance
            box_width = x2 - x1
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)

            # Object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_PLAIN
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            # Display class name
            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

            # Display confidence
            accuracy_text = f'Accuracy: {confidence}'
            cv2.putText(img, accuracy_text, (x1, y1 - 10), font, fontScale, color, thickness)

            # Display distance in meters
            distance_text = f'Distance: {distance:.2f} meters'
            cv2.putText(img, distance_text, (x1, y2 + 20), font, fontScale, color, thickness)

            # Voice feedback
            voice_feedback = f"{class_name} detected with {confidence} accuracy. Distance is {distance:.2f} meters."
            engine.say(voice_feedback)
            engine.runAndWait()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
