from kivy.lang import Builder
from plyer import gps
from kivy.app import App
from kivy.properties import StringProperty
from kivy.clock import Clock, mainthread
from kivy.uix.boxlayout import BoxLayout

import cv2
import torch
import numpy as np
import requests
from torchvision import transforms
from PIL import Image as im
import random


kv = '''
<MyLayout>:
    label1: label1
    orientation: 'vertical'

    Label:
        text: 'drowsiness'
        id: label1

    Button:
        text: "start monitoring"
        font_size: 32
        size: 100,50
        on_press:root.start()
'''


model6 = "DMS_mobilevitv2_100.pt"
AWB = True

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class_names = ['drowsy', 'focused',  'using phone', 'yawning']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Load the TorchScript model (self-contained .pt file generated earlier)
    model = torch.jit.load(model6, map_location=device)
    model.to(device)
    model.eval()
except FileNotFoundError:
    exit()
transform = transforms.Compose([ transforms.Resize((224, 224)), 
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])


def detect_and_crop_face(frame):
    """ Detects face in frame and crops. Also returns ]coordinates for bounding box. If no face, returns None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face_crop = frame[y:y+h, x:x+w]  # Crop face region (remains in RGB)
        return face_crop, (x, y, w, h)
    
    return None, None




Builder.load_string(kv)
class MyLayout(BoxLayout):

    capture = cv2.VideoCapture()

    def start(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.loadVid , 1)
        
    def loadVid(self, *args):
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

        ret, frame = self.capture.read()
        if not ret:
            pass
        face, face_coords = detect_and_crop_face(frame)

        if face is not None:
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = im.fromarray(rgb_face)

            input_tensor = transform(pil_image).unsqueeze(0).to(device)

                # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert logits to probabilities
                confidence, predicted = torch.max(probabilities, 0)  # Get highest confidence prediction
                predicted_label = class_names[predicted.item()]
                self.label1.text = predicted_label



class scenes(App):
    def build(self):
        return MyLayout()

tester = scenes()
tester.run()

