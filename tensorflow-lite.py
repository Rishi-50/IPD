import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the minimum confidence threshold for detections
min_conf_threshold = 0.5

# Define labels for the classes
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
          "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
          "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
          "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
          "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
          "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
          "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Load the input image and resize it to match the input tensor shape
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input image
    input_data = cv2.resize(frame, (300, 300))
    input_data = input_data[:, :, [2, 1, 0]]  # BGR to RGB
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5  # Normalize

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Perform inference
    interpreter.invoke()

    # Get the output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    # Draw bounding boxes on the input image
    for i in range(num_detections):
        if scores[0, i] > min_conf_threshold:
            class_id = int(classes[0, i])
            box = boxes[0, i]
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            # Convert bounding box coordinates to actual pixel values
            (left, right, top, bottom) = (xmin * 640, xmax * 640, ymin * 480, ymax * 480)

            # Draw bounding box and label on the image
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=2)
            cv2.putText(frame, labels[class_id], (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
