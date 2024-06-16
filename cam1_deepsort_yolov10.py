import os
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import torch
import datetime
import logging
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_camera(serial_number):
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)
        logger.info(f"Camera with serial number {serial_number} configured successfully.")
        return pipeline, profile
    except Exception as e:
        logger.error(f"Failed to configure camera with serial number {serial_number}: {str(e)}")
        raise

def initialize_model():
    model_path = "/home/objectdetection/yolov10/runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found at {model_path}")
        raise FileNotFoundError("Model weights file not found")
    
    model = YOLOv10(model_path)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model

def load_class_names():
    classes_path = "/home/objectdetection/yolov10/pipeline/class.names"
    if not os.path.exists(classes_path):
        logger.error(f"Class names file not found at {classes_path}")
        raise FileNotFoundError("Class names file not found")
    
    with open(classes_path, "r") as f:
        file_content = f.read().strip()
        logger.info(f"File content: {file_content}")
        class_names = file_content.split("\n")
    
    logger.info(f"Loaded class names: {class_names}")
    return class_names

def process_frame(frame, model, tracker, class_names, colors, conf_threshold=0.5):
    results = model(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)
        
        if confidence < conf_threshold:
            continue
        
        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
    
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

def draw_tracks(frame, tracks, class_names, colors, class_counters, track_class_mapping, sent_ids):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        
        if class_id >= len(class_names):
            logger.warning(f"Invalid class_id {class_id}, skipping.")
            continue
        
        x1, y1, x2, y2 = map(int, ltrb)
        color = colors[class_id % len(colors)]
        B, G, R = map(int, color)

        # Assign a new class-specific ID if the track_id is seen for the first time
        if track_id not in track_class_mapping:
            class_counters[class_id] += 1
            track_class_mapping[track_id] = class_counters[class_id]
        
        class_specific_id = track_class_mapping[track_id]
        text = f"{class_specific_id} - {class_names[class_id]}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if track_id not in sent_ids:
            frame = send_location(frame, track_id, (x1, y1, x2, y2))
            sent_ids.add(track_id)
    
    return frame

def send_location(frame, track_id, bbox):
    logger.info(f"Sending location for track ID {track_id}: {bbox}")
    
    frame_height, frame_width = frame.shape[:2]
    box_width, box_height = 200, 50
    top_left = (frame_width // 2 - box_width // 2, 10)
    bottom_right = (frame_width // 2 + box_width // 2, 10 + box_height)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), -1)
    cv2.putText(frame, "Location Sent", (top_left[0] + 10, top_left[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def capture_frames(pipeline, model, tracker, class_names, colors, class_counters, track_class_mapping, sent_ids):
    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            tracks = process_frame(color_image, model, tracker, class_names, colors)
            color_image = draw_tracks(color_image, tracks, class_names, colors, class_counters, track_class_mapping, sent_ids)

            cv2.imshow("Object Detection", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print(f"Capture thread finished.")
        cv2.destroyAllWindows()

def run_camera(pipeline, model, tracker, class_names, colors, class_counters, track_class_mapping, sent_ids):
    thread = threading.Thread(target=capture_frames, args=(pipeline, model, tracker, class_names, colors, class_counters, track_class_mapping, sent_ids))
    thread.start()
    return thread

if __name__ == "__main__":
    serial_number = "234222302429"  # Replace with your camera's serial number

    try:
        pipeline, profile = configure_camera(serial_number)
    except Exception as e:
        logger.error(f"Error configuring camera with serial number {serial_number}: {str(e)}")
        pipeline = None

    if pipeline:
        model = initialize_model()
        tracker = DeepSort(max_age=30)
        class_names = load_class_names()
        colors = np.random.uniform(0, 255, size=(len(class_names), 3))
        class_counters = defaultdict(int)
        track_class_mapping = {}
        sent_ids = set()

        run_camera(pipeline, model, tracker, class_names, colors, class_counters, track_class_mapping, sent_ids)
    else:
        logger.error("Failed to configure the camera.")
