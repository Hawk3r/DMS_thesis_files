import cv2
import torch
import timm
import numpy as np
import requests
from torchvision import transforms
from PIL import Image

URL = "http://192.168.0.128"
AWB = True

# Load OpenCV's built-in face detector  ,# Example: 6 classes for driver monitoring
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
num_classes = 6; class_names = ['DangerousDriving', 'Distracted', 'Drinking', 'SafeDriving', 'SleepyDriving', 'Yawn']

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnet_lite0', pretrained=False)
model.reset_classifier(num_classes)
model.load_state_dict(torch.load("best_efficientnet_lite_model.pth", map_location=device))
model.to(device)
model.eval()

# Define the same transforms used during training/validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure size matches training data
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
    ])

def simulate_ir_effect(frame):
    """Converts an RGB frame to a simulated infrared-like grayscale image.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)  # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def detect_and_crop_face(frame):
    """Detects a face in the frame and crops it.
    Also returns face coordinates (x, y, w, h) for drawing a bounding box.  If no face is found, returns None."""
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face_crop = frame[y:y+h, x:x+w]  # Crop face region
        return face_crop, (x, y, w, h)
    return None, None

def main():
    # Open webcam on index 2 (change if needed)
    cap = cv2.VideoCapture("http://192.168.0.128:81/stream")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)      # Lower resolution to match model's expected input size
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

    while True:
        ret, frame = cap.read()
        if not ret:     break
        ir_like_frame = simulate_ir_effect(frame)
        face, face_coords = detect_and_crop_face(ir_like_frame)
        cv2.imshow("Driver ", ir_like_frame)

        if face is not None: # Transform for model input
            rgb_face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(rgb_face)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert logits to probabilities
                confidence, predicted = torch.max(probabilities, 0)  # Get highest confidence prediction
                predicted_label = class_names[predicted.item()]
                print(predicted_label)

            # Draw bounding box on the face
            x, y, w, h = face_coords
            cv2.rectangle(ir_like_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)  # White bounding box
            cv2.putText(ir_like_frame, f'{predicted_label} ({confidence:.2f})', 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

        # Show only ONE camera view with face detection and classification
        cv2.imshow("Driver Monitoring", ir_like_frame)
        
        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()