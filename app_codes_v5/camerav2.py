import cv2
import torch
import timm
import numpy as np
import requests
from torchvision import transforms
from PIL import Image

URL = "http://192.168.4.1"
AWB = True


# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
num_classes = 6  # Adjusted for the new number of classes
class_names = ['drowsy', 'face', 'focused', 'holding phone', 'yawning', 'Unknown']

# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnet_lite0', pretrained=False)
model.reset_classifier(num_classes)

try:
    model.load_state_dict(torch.load("best_efficientnet_lite_model_ownmodel.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Please check the path.")
    exit()

# Define the same transforms used during training/validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure size matches training data
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet normalization
                         [0.229, 0.224, 0.225])
])


def detect_and_crop_face(frame):
    """
    Detects a face in the frame and crops it.
    Also returns face coordinates (x, y, w, h) for drawing a bounding box.
    If no face is found, returns None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection only
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face_crop = frame[y:y+h, x:x+w]  # Crop face region (remains in RGB)
        return face_crop, (x, y, w, h)
    
    return None, None


def main():
    # Open webcam on index 2 (change if needed)
    cap = cv2.VideoCapture(URL +":81/stream")
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
            pil_image = Image.fromarray(rgb_face)

            # Transform for model input
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert logits to probabilities
                confidence, predicted = torch.max(probabilities, 0)  # Get highest confidence prediction
                predicted_label = class_names[predicted.item()]

            # Draw bounding box on the face
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)  # White bounding box
            cv2.putText(frame, f'{predicted_label} ({confidence:.2f})', 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show only ONE camera view with face detection and classification
        cv2.imshow("Driver Monitoring", frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
