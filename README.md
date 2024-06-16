# Realsense Yolov10 Deepsort Realtime Pipeline

This repository contains code for an object detection and tracking system using a RealSense camera, YOLOv10 for object detection, and DeepSORT for tracking. The system captures frames from the camera, processes them to detect objects, and tracks the detected objects over time.

### Prerequisites
- Python 3.7 or higher
- Required Libraries:
- OpenCV
- NumPy
- Pyrealsense2
- Torch
- DeepSort
- YOLOv10

### Installing Dependencies
You can install the necessary dependencies using pip:</br>

```pip install opencv-python numpy pyrealsense2 torch torchvision deepsort-realtime ultralytics```

## Usage
- Step 1: Configure the Camera</br>
Ensure that your RealSense camera is connected to your system and note its serial number. The serial number is used to configure the camera.

- Step 2: Model and Class Names</br>
Ensure that the model weights and class names file are in the specified paths:</br>

Model weights file: /path/weights/best.pt</br>
Class names file: /path/pipeline/class.names

- Step 3: Run the Script
Run the script by executing the following command:</br>

```python main.py```

Stopping the Script
Press q to stop the script and close the display window.

Configuration
Camera Configuration
The configure_camera function configures the RealSense camera with the specified serial number and sets up the video stream parameters.

python
Copy code
def configure_camera(serial_number):
    ...
Model Initialization
The initialize_model function loads the YOLOv10 model with the specified weights.

python
Copy code
def initialize_model():
    ...
Class Names Loading
The load_class_names function reads the class names from a specified file.

python
Copy code
def load_class_names():
    ...
Frame Processing
The process_frame function processes each frame to detect objects using the YOLOv10 model.

python
Copy code
def process_frame(frame, model, tracker, class_names, colors, conf_threshold=0.5):
    ...
Drawing Tracks
The draw_tracks function draws the bounding boxes and labels for tracked objects on the frame.

python
Copy code
def draw_tracks(frame, tracks, class_names, colors, class_counters, track_class_mapping, sent_ids):
    ...
Capturing Frames
The capture_frames function captures frames from the RealSense camera and processes them in a loop.

python
Copy code
def capture_frames(pipeline, model, tracker, class_names, colors, class_counters, track_class_mapping, sent_ids):
    ...
Running the Camera
The run_camera function starts a separate thread to capture and process frames.

python
Copy code
def run_camera(pipeline, model, tracker, class_names, colors, class_counters, track_class_mapping, sent_ids):
    ...
Features
Object Detection: Uses YOLOv10 for detecting objects in the frames.
Object Tracking: Tracks detected objects across frames using DeepSORT.
Camera Integration: Captures real-time video feed from a RealSense camera.
Logging: Provides detailed logs for each step of the process.
Logging
Logging is configured to provide detailed information about the process. Logs are printed to the console.

python
Copy code
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Troubleshooting
Camera Not Found: Ensure the RealSense camera is connected and the serial number is correct.
Model Weights Not Found: Verify the model weights file path.
Class Names File Not Found: Verify the class names file path.
Dependencies: Ensure all required libraries are installed.
