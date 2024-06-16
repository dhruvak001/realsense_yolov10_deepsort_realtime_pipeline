# Realsense Yolov10 Deepsort Realtime Pipeline

This repository contains code for an object detection and tracking system using a RealSense camera, YOLOv10 for object detection, and DeepSORT for tracking. The system captures frames from the camera, processes them to detect objects, and tracks the detected objects over time.

## Introduction
This system leverages state-of-the-art deep learning models and tracking algorithms to achieve robust and accurate object detection and tracking. The main components of the system are:

- YOLOv10: A highly efficient and accurate object detection model.
- DeepSORT: A deep learning-based object tracking algorithm.
- RealSense Camera: Provides high-quality video streams for real-time processing.
- The goal of the system is to detect objects in real-time from the video stream provided by the RealSense camera and to track these objects across frames, providing consistent identifiers for each tracked object.

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


- Step 3: Run the Script
Run the script by executing the following command:</br>
```python main.py```

- Stopping the Script</br>
Press q to stop the script and close the display window.

## Features
- Object Detection: Uses YOLOv10 for detecting objects in the frames.
- Object Tracking: Tracks detected objects across frames using DeepSORT.
- Camera Integration: Captures real-time video feed from a RealSense camera.
- Logging: Provides detailed logs for each step of the process.
- 
### Logging
Logging is configured to provide detailed information about the process. Logs are printed to the console.


```logging.basicConfig(level=logging.INFO)```
```logger = logging.getLogger(__name__)```

### Troubleshooting
- Camera Not Found: Ensure the RealSense camera is connected and the serial number is correct.
- Model Weights Not Found: Verify the model weights file path.
- Class Names File Not Found: Verify the class names file path.
- Dependencies: Ensure all required libraries are installed.
