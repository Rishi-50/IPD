import cv2
import numpy as np
import onnxruntime
from picamera.array import PiRGBArray
from picamera import PiCamera

# Load the ONNX model
onnx_model_path = "yolov4.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Function to preprocess the image
def preprocess(frame):
    resized = cv2.resize(frame, (416, 416))
    normalized = resized.astype(np.float32) / 255.0
    return np.transpose(normalized, [2, 0, 1]).reshape(1, 3, 416, 416)

# Function to postprocess the output
def postprocess(outputs, frame):
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_ids[i]}", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Initialize the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # Grab the raw NumPy array representing the image
    image = frame.array
    
    # Perform inference
    input_data = preprocess(image)
    outputs = session.run(output_names, {input_name: input_data})
    result_frame = postprocess(outputs, image)
    
    # Display the result
    cv2.imshow("YOLOv4 Object Detection", result_frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Clear the stream for the next frame
    raw_capture.truncate(0)
    
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
