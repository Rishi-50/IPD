import numpy as np
import pyttsx3
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

def get_bboxes(image, path_to_onnx): 
    YOLOWEIGHTS = path_to_onnx
    model = cv2.dnn.readNetFromONNX(YOLOWEIGHTS)
    
    CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

    original_image = image
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=False)
    model.setInput(blob)
    outputs = model.forward()
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    boxes = []
    scores = []
    class_ids = []
    
    # Loop through the outputs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.2:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    
    conf_threshold = 0.5  # Confidence threshold
    iou_threshold = 0.45   # IOU threshold
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold,iou_threshold, 0.5)
    
    detections = []
    classes_count = {}  # Dictionary to store class counts

    # Loop through the result boxes
    for i in range(len(result_boxes)):
        index = result_boxes[i][0]
        class_id = class_ids[index]
        class_name = CLASSES[class_id]

        if class_name in classes_count:
            classes_count[class_name] += 1
        else:
            classes_count[class_name] = 1

        detection = {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': scores[index],
            'box': boxes[index],
            'scale': scale
        }
        detections.append(detection)
        
    return classes_count, detections

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    raw_capture = PiRGBArray(camera, size=(640, 480))

    time.sleep(0.1)

    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array

        classes_count, detections = get_bboxes(image, "yolov8n.onnx")
        
        for detection in detections:
            class_name = detection['class_name']
            count = classes_count[class_name]
            text_to_speech(f"{count} {class_name}")

        # Draw bounding boxes on the frame
        for detection in detections:
            box = detection['box']
            x, y, w, h = [int(coord) for coord in box]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, detection['class_name'], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Object Detection', image)
        key = cv2.waitKey(1) & 0xFF

        raw_capture.truncate(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
