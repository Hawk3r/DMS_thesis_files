#non kivy imports
import cv2
import torch
import numpy as np
import requests
from torchvision import transforms
from PIL import Image as im
import random


model1 = "DMS_efficientnet_lite0.pt"
model2 = "DMS_efficientnetv2_rw_t.pt"
model3 = "DMS_mobilenetv3_large_100.pt"
model4 = "DMS_mobilenetv3_small_100.pt"
model5 = "DMS_mobilevit_xs.pt"
model6 = "DMS_mobilevitv2_100.pt"
AWB = True

# Load OpenCV's built-in face detector  ,6 classes for driver monitoring
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
num_classes =4
class_names = ['drowsy', 'focused',  'using phone', 'yawning']
# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Load the TorchScript model (self-contained .pt file generated earlier)
    model = torch.jit.load(model6, map_location=device)
    model.to(device)
    model.eval()
    print("TorchScript model loaded successfully.")
except FileNotFoundError:
    print("Error: TorchScript model file not found. Please check the path.")
    exit()
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure size matches training data
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
    ])


def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection only
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face_crop = frame[y:y+h, x:x+w]  # Crop face region (remains in RGB)
        return face_crop, (x, y, w, h)
    
    return None, None


def main():
    # Open webcam on index 2 (change if needed)
    #cap = cv2.VideoCapture("http://192.168.4.1:81/stream")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Lower resolution to match model's expected input size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and crop face, also get bounding box coordinates
        face, face_coords = detect_and_crop_face(frame)

        if face is not None:
            # Convert face from BGR (OpenCV format) to RGB (PIL format)
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = im.fromarray(rgb_face)

            # Transform for model input
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert logits to probabilities
                confidence, predicted = torch.max(probabilities, 0)  # Get highest confidence prediction
                predicted_label = class_names[predicted.item()]
                print(predicted_label)


if __name__ == "__main__":
    main()
