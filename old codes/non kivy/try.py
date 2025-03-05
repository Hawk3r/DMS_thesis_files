import cv2
import numpy as np
import time
import threading
import queue

# Load MobileNet SSD (lightweight face detection)
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Enable GPU acceleration (if available)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)  # Change to CUDA if using Nvidia

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Multi-threading setup
frame_queue = queue.Queue(maxsize=1)
processed_frame = None
detecting = False
frame_skip = 3  # Process every 3rd frame
frame_id = 0

def detect_faces():
    global processed_frame, detecting
    while True:
        if not frame_queue.empty() and not detecting:
            detecting = True
            frame = frame_queue.get()
            h, w = frame.shape[:2]

            # Resize for faster processing (160x120)
            small_frame = cv2.resize(frame, (160, 120))

            # Convert to blob and process
            blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            # Draw detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            processed_frame = frame
            detecting = False

# Start detection thread
thread = threading.Thread(target=detect_faces, daemon=True)
thread.start()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip == 0:  # Process every 3rd frame
        if not frame_queue.full():
            frame_queue.put(frame.copy())

    # Display processed frame
    display_frame = processed_frame if processed_frame is not None else frame

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Ultra-Fast Driver Monitoring - Head Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
