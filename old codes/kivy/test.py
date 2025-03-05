import cv2
import torch
import timm
import numpy as np
import requests
from torchvision import transforms
from PIL import Image
import random

import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock





URL = "http://192.168.68.124"
AWB = True

# Load OpenCV's built-in face detector  ,# Example: 6 classes for driver monitoring
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
num_classes = 6
class_names = ['DangerousDriving', 'Distracted', 'Drinking', 'SafeDriving', 'SleepyDriving', 'Yawn']

# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnet_lite0', pretrained=False)
model.reset_classifier(num_classes)
model.load_state_dict(torch.load("best_efficientnet_lite_model.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),  transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet normalization
                         [0.229, 0.224, 0.225])                         ])


#cap = cv2.VideoCapture("http://192.168.0.128:81/stream")
class Myroot(BoxLayout):
    def __init__(self):
        super(Myroot, self).__init__()
    
    def generate_num(self): self.random_label.text = str(random.randint(0,100))

    def loadVid(self,*args):
        ret, frame = self.capture.read()
        self.image_frame = frame
        buffer= cv2.flip(frame, 0).tostring()
        texture = Texture.create(size = (frame.shape[1], frame.shape[0]),colorfmt = 'bgr')
        texture.blit_buffer(buffer, colorfmt = 'bgr',bufferfmt = 'ubyte')
        self.image.texture = texture

class tester(App):
    def build(self):
        return Myroot()
    

tester = tester()
tester.run()