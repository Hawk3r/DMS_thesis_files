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


def get_current_gps_coordinates():
    g = geocoder.ip('me')#this function is used to find the current information using our IP Add
    if g.latlng is not None: #g.latlng tells if the coordiates are found or not
        return g.latlng
    else:
        return None





class LoginScreen(Screen):
    def login(self):
        username = self.logname.text
        pword = self.password.text
        if username == pword:
            self.manager.current = 'menu'
    
class MenuScreen(Screen):
    capture = cv2.VideoCapture()
    conversation = []
    activated = False  # Activation flag
    api_key = ""  # Replace with your actual API key # Hardcoded API Key for Testing
    client = genai.Client(api_key= api_key)# Initialize Gemini client
    timer = 0

    def start(self): 
        URL  = self.inp.text
        
        self.capture = cv2.VideoCapture(URL + ":81/stream")
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

        coordinates = get_current_gps_coordinates()
        if coordinates is not None:
            latitude, longitude = coordinates

            self.gpslabel.text = "Latitude: " + str(latitude) + "Longitude: "+ str(longitude)

        else:
            self.gpsgpslabel.text = "Unable to retrieve your GPS coordinates."
    
    def send_message(self):
        """ Sends the spoken or typed message to the chatbot. """
        text =  self.ids.user_input.text.strip()
        

        if not text:
            return
        
        if not self.activated:
            if text.lower() == "hello dms":
                self.activated = True
                self.conversation.append({"role": "user", "content": text})

                # Stronger first message to enforce driver monitoring
                system_instruction = ("You are the **Driver Monitoring Assistant (DMS)**. "
                    "Your job is to **detect drowsiness, check driver alertness, and ensure road safety**. "
                    "If the driver reports feeling tired, recommend stopping for rest or drinking coffee. "
                    "Ask how long they have been driving and give advice based on their response. "
                    "Keep responses **short and direct**.")

                threading.Thread(target=self.call_gemini_api, args=(system_instruction,)).start()
            else:
                self.add_chat_message("DMS", "Please say 'Hello DMS' to activate the system.")
        else:
            self.conversation.append({"role": "user", "content": text})
            threading.Thread(target=self.call_gemini_api, args=(text,)).start()

    

    def add_chat_message(self, sender, message):
        msg = f"{sender}: {message}"

        self.chatout.text = msg
        
    


    def call_gemini_api(self, user_input):
        """ Calls the Gemini API using the google.genai client. """
        try:
            modl = "gemini-2.0-flash"
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="(IMPORTANT: Stay in role. You are the Driver Monitoring Assistant. "
                                 "Do NOT talk about yourself. ONLY answer as a driver safety system.)\n\n"
                                 + user_input
                        ),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_output_tokens=500,
                response_mime_type="text/plain",
            )

            response_text = ""

            for chunk in self.client.models.generate_content_stream(
                model=modl,
                contents=contents,
                config=generate_content_config,
            ):
                response_text += chunk.text

            reply = response_text if response_text else "No response from Gemini."
        except Exception as e:
            reply = f"Exception: {e}"

        self.conversation.append({"role": "model", "content": reply})
        Clock.schedule_once(lambda dt: self.add_chat_message("DMS", reply))

    

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
