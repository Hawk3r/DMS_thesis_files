#added ai chatbot, will improve upon it in v6


from kivy.lang import Builder
from plyer import gps
from kivy.app import App
from kivy.properties import StringProperty
from kivy.clock import Clock, mainthread


import cv2
import torch
import numpy as np
import requests
from torchvision import transforms
from PIL import Image as im
import random
import time
from plyer import tts
import json
kv = '''
BoxLayout:
    orientation: 'vertical'

    Label:
        text: app.gps_location

    Label:
        text: app.gps_status
    
    Label:
        text: app.drows
    
    Label:
        text: app.time
    
    BoxLayout:
        Label:
            text: app.d1
        Label:
            text: app.d2
        Label:
            text: app.d3
        Label:
            text: app.finl


    BoxLayout:
        size_hint_y: None
        height: '48dp'
        padding: '4dp'

        ToggleButton:
            text: 'Start' if self.state == 'normal' else 'Stop'
            on_state:
                app.start(1000, 0) if self.state == 'down' else \
                app.stop()
        Button:
            text: 'monitor'
            on_press: app.dd()
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

""" Detects face in frame and crops. Also returns ]coordinates for bounding box. If no face, returns None."""
def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face_crop = frame[y:y+h, x:x+w]  # Crop face region (remains in RGB)
        return face_crop, (x, y, w, h)
    return None, None



class GeminiChatbot:
    def __init__(self, api_key, model="gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.chat_history = []  # Proper Gemini format: list of dicts with 'role' and 'parts'

    def send_message(self, user_message):
        # Add the user message to history
        self.chat_history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        data = {
            "contents": self.chat_history
        }

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()

            result = response.json()
            reply = result["candidates"][0]["content"]["parts"][0]["text"]

            # Add model's reply to history
            self.chat_history.append({
                "role": "model",
                "parts": [{"text": reply}]
            })

            return reply

        except requests.exceptions.RequestException as e:
            return f"Network error: {e}"
        except Exception as e:
            return f"Error: {e}"
        








class GpsTest(App):
    capture = cv2.VideoCapture()
    gps_location = StringProperty()
    gps_status = StringProperty('Click Start to get GPS location updates')
    drows = StringProperty("drows")
    time = StringProperty("latency")
    finl = StringProperty("final status")
    bot = GeminiChatbot("AIzaSyDU3X8MnBiFBvART6R9zMeyuL_UCHeLXTI")

    drowsCounter = 0
    counter = 0
    distcounter = 0
    drowsthresh = 10
    distthresh = 10
    
    msg= False

    d1 =StringProperty()
    d2 =StringProperty()
    d3 =StringProperty()

    instances_of_drowsiness = 0
    c= 0
    llmcounter = 0

    def build(self):
        try:
            gps.configure(on_location=self.on_location, on_status=self.on_status)
        except NotImplementedError:
            import traceback
            traceback.print_exc()
            self.gps_status = 'GPS is not implemented for your platform'

        return Builder.load_string(kv)

    def start(self, minTime, minDistance):
        
        gps.start(minTime, minDistance)


    def stop(self):
        gps.stop()

    def dd(self):
        tts.speak(self.bot.send_message("You are the **Driver Monitoring Assistant (DMS)**. "
                    "Your job is to **detect drowsiness, check driver alertness, and ensure road safety**. "
                    "If the driver reports feeling tired, recommend stopping for rest or drinking coffee. "
                    "Ask how long they have been driving and give advice based on their response. "
                    "Keep responses ** short and direct**."))
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.loadVid , 1/12)

    
    def on_location(self, **kwargs):
        self.latitude = kwargs.get('lat')
        self.longitude = kwargs.get('lon')
        self.speed = kwargs.get('speed')

        self.gps_location = f'Latitude: {self.latitude}\nLongitude: {self.longitude}\nSpeed: {self.speed}'

    
    def on_status(self, stype, status):
        self.gps_status = 'type={}\n{}'.format(stype, status)

    def on_pause(self):
        gps.stop()
        return True

    def on_resume(self):
        gps.start(1000, 0)
        pass
    

    def loadVid(self, *args):
        start = time.time()
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
                self.drows = predicted_label


                if predicted_label == "focused":
                    self.counter = self.counter +1           
                    if self.counter > 4:
                        self.msg = False
                        self.finl ="FOCUSED"
                        self.distcounter = 0
                        self.llmcounter = 0
                        self.drowsCounter = 0
                        if self.counter >20:
                            self.counter =5
                
                #drowsy       
                elif (predicted_label == "drowsy" )or (predicted_label == "sleepy") or ((predicted_label == "yawning")):
                    self.drowsCounter = self.drowsCounter+1
                    self.counter = 0
                    if self.drowsCounter > self.drowsthresh:
                        self.finl ="SLEEPY"
                        if not self.msg:
                            tts.speak(self.bot.send_message("im sleepy"))
                            self.msg = True
                        if self.drowsCounter >50:
                            self.drowsCounter =  self.drowsthresh+ 1
                        if self.llmcounter == 0:
                            self.llmcounter = 1

                #distracted
                else:
                    self.distcounter = self.distcounter +1
                    if self.distcounter >self.distthresh:
                        self.counter = 0
                        self.finl ="DISTRACTED"
                        if not self.msg:
                            tts.speak(self.bot.send_message("im distracted"))
                            self.msg = True
                        if self.llmcounter == 0:
                            self.llmcounter = self.distthresh+ 1
                        #if self.distcounter >10:
                        #    self.distcounter = 11

        end = time.time()
        self.time = str(end - start)

        self.d1 = "counter " +str(self.counter)  
        self.d2 = "drowscounter "+str(self.drowsCounter)
        self.d3 = "distcounter "+str(self.distcounter)
        


if __name__ == '__main__':
    GpsTest().run()
