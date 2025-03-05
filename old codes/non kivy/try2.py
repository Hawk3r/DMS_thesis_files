import cv2
import torch
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load YOLOv7-Tiny model for face detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load EfficientNetV2 for eye state detection
efficientnet = EfficientNetV2S(weights="imagenet")

# Start video capture (lower resolution for speed)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize FPS counter
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster YOLO processing
    small_frame = cv2.resize(frame, (320, 240))

    # Run YOLOv7 face detection
    results = model(small_frame)
    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = int(x1 * 2), int(y1 * 2), int(x2 * 2), int(y2 * 2)  # Scale back

            # Crop detected face
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocess for EfficientNetV2 (fixed input size)
            face_resized = cv2.resize(face, (384, 384))
            face_resized = np.expand_dims(face_resized, axis=0)
            face_resized = preprocess_input(face_resized)

            # Predict drowsiness
            preds = efficientnet.predict(face_resized)
            label = "Drowsy" if np.argmax(preds) in [1, 2] else "Awake"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Driver Monitoring (YOLOv7-Tiny + EfficientNetV2)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()