import cv2
import torch
import timm
import numpy as np
import requests
from torchvision import transforms
from PIL import Image as im
import random
import kivy
import kivymd
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivymd.uix.textfield import textfield
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.slider import Slider

import geocoder


import os
import threading
import base64
import speech_recognition as sr  # Speech recognition for voice input
from google import genai
from google.genai import types



AWB = True

# Load OpenCV's built-in face detector  ,# Example: 6 classes for driver monitoring
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
num_classes = 6  # Adjusted for the new number of classes
class_names = ['drowsy', 'face', 'focused', 'holding phone', 'yawning', 'Unknown']

# Load trained model
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure size matches training data
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
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





class LoginScreen(Screen):
    def login(self):
        username = self.logname.text
        pword = self.password.text
        if username == pword:
            self.manager.current = 'menu'
    
class MenuScreen(Screen):
    drowsCounter = 0
    counter = 0
    capture = cv2.VideoCapture()
    conversation = []
    activated = False  # Activation flag
    
    timer = 0

    def start(self): 
        
        
        self.capture = cv2.VideoCapture("http://192.168.4.1:81/stream")
        Clock.schedule_interval(self.loadVid , 1.0/24.0)

    def loadVid(self,*args):
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        ret, frame = self.capture.read()

        
        face, face_coords = detect_and_crop_face(frame)
        
        if face is not None: # Transform for model input
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = im.fromarray(rgb_face)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert logits to probabilities
                confidence, predicted = torch.max(probabilities, 0)  # Get highest confidence prediction
                predicted_label = class_names[predicted.item()]
                print(predicted_label)
                self.testlabel.text =predicted_label

                if predicted_label == "focused":
                    self.counter = self.counter +1
                    self.count1.text = "counter " +str(self.counter)
                    
                    if self.counter > 12:
                        self.gpslabel.text ="FOCUSED"
                        self.drowsCounter = 0
                        self.count2.text = "drwoscounter "+str(self.drowsCounter)
                        
                else:
                    self.drowsCounter = self.drowsCounter+1
                    self.counter = 0
                    self.count2.text ="drwoscounter "+ str(self.drowsCounter)
                    
                    self.count1.text ="counter " + str(self.counter)
                    if self.drowsCounter > 48:
                        self.gpslabel.text ="DISTRACTED"
                

            self.label.text = str(self.timer)
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2) 
            cv2.putText(frame, f'{predicted_label} ({confidence:.2f})', 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

   
    
        self.image_frame = frame
        buffer= cv2.flip(frame, 0).tostring()
        texture = Texture.create(size = (frame.shape[1], frame.shape[0]),colorfmt = 'bgr')
        texture.blit_buffer(buffer, colorfmt = 'bgr',bufferfmt = 'ubyte')
        self.image.texture = texture
        



class SettingsScreen(Screen):
    pass


class ChatScreen(Screen):
    pass


class scenes(App):

    def build(self):
        # Create the screen manager
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(SettingsScreen(name='settings'))
        sm.add_widget(ChatScreen(name='chat'))


        return sm


tester = scenes()
tester.run()
