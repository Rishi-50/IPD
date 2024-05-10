# Import necessary libraries
import time
import numpy as np
import cv2
import mediapipe as mp
from picamera.array import PiRGBArray
from picamera import PiCamera

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

def run(model: str, max_results: int, score_threshold: float, 
        width: int, height: int) -> None:
    """Continuously run inference on images acquired from the PiCamera.

    Args:
      model: Name of the TFLite object detection model.
      max_results: Max number of detection results.
      score_threshold: The score threshold of detection results.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
    """

    # Initialize PiCamera
    camera = PiCamera()
    camera.resolution = (width, height)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(width, height))

    # Wait for the camera to warm up
    time.sleep(0.1)

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None
    detection_result_list = []

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        detection_result_list.append(result)
        COUNTER += 1

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           max_results=max_results, score_threshold=score_threshold,
                                           result_callback=save_result)
    detector = vision.ObjectDetector.create_from_options(options)

    # Continuously capture images from the PiCamera and run inference
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # Grab the raw NumPy array representing the image
        image = frame.array

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if detection_result_list:
            # print(detection_result_list)
            current_frame = visualize(current_frame, detection_result_list[0])
            detection_frame = current_frame
            detection_result_list.clear()

        if detection_frame is not None:
            cv2.imshow('object_detection', detection_frame)

        # Clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
#      default='efficientdet_lite0.tflite')
        default='best.tflite')
    parser.add_argument(
        '--maxResults',
        help='Max number of detection results.',
        required=False,
        default=5)
    parser.add_argument(
        '--scoreThreshold',
        help='The score threshold of detection results.',
        required=False,
        type=float,
        default=0.25)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    args = parser.parse_args()

    run(args.model, int(args.maxResults),
        args.scoreThreshold, args.frameWidth, args.frameHeight)

if __name__ == '__main__':
    main()
