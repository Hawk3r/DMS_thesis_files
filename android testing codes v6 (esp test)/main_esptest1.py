import numpy as np
import cv2
from urllib.request import urlopen
import tkinter as tk
import time
import torch
import requests
from torchvision import transforms
from PIL import Image as im
import random


AWB = True

# Load OpenCV's built-in face detector  ,6 classes for driver monitoring
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
num_classes =4
class_names = ['drowsy', 'focused',  'using phone', 'yawning']
# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
try:
    # Load the TorchScript model (self-contained .pt file generated earlier)
    model = torch.jit.load("DMS_mobilevitv2_100.pt", map_location=device)
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
'''



def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    image = cv2.resize(image, (256,256))

    return image




def detect_and_crop_face(frame):
    """
    Detects a face in the frame and crops it. Also returns face coordinates (x, y, w, h) for drawing a bounding box. If no face is found, returns None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection only
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face_crop = frame[y:y+h, x:x+w]  # Crop face region (remains in RGB)
        return face_crop, (x, y, w, h)
    
    return None, None



def main():
    frame = url_to_image("http://192.168.4.1/capture")

        # Detect and crop face, also get bounding box coordinates
    face, face_coords = detect_and_crop_face(frame)
    print("a")

    if face is not None:
        print("face detected")


print("initialize")
while True:
    main()

