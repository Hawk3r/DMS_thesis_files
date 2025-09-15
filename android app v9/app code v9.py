from kivy.lang import Builder
from plyer import gps
from kivy.app import App
from kivy.properties import StringProperty
from kivy.clock import Clock, mainthread
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.slider import Slider
from plyer import tts

import cv2
import torch
import numpy as np
import requests
from torchvision import transforms
from PIL import Image as im
import random
import time

from urllib.request import urlopen
import json


# ── FIREBASE INTEGRATION ───────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

cred = credentials.Certificate("serviceAccount.json")  # path to your Firebase key
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── FIREBASE HELPERS ──────────────────────────────────
def add_user_account(username, password, first_name, last_name, car_brand, car_model):
    users_ref = db.collection("useracc")
    existing = users_ref.where("username", "==", username).get()

    if not existing:
        users_ref.add({
            "username": username,
            "password": password,  # ⚠️ In production, hash passwords
            "created_at": datetime.utcnow().isoformat()
        })
        db.collection("drivers").add({
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"{first_name} {last_name}".title(),
            "car_brand": car_brand,
            "car_model": car_model,
            "created_at": datetime.utcnow().isoformat()
        })
        print(f"✅ User {username} registered")
        return True
    else:
        return False

def validate_user(username, password):
    users_ref = db.collection("useracc")
    query = users_ref.where("username", "==", username).where("password", "==", password).get()
    return len(query) > 0

def get_driver_full_name(username):
    drivers = db.collection("drivers").where("username", "==", username).get()
    if drivers:
        return drivers[0].to_dict().get("full_name")
    return None

def log_to_firestore(driver, label, latitude=None, longitude=None):
    db.collection("driver_logs").add({
        "driver": driver,
        "label": label,
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": datetime.utcnow().isoformat()
    })

# ── KIVY LAYOUT ───────────────────────────────────────

kv = '''
ScreenManager:
    LoginScreen:
    RegisterScreen:
    MainScreen:

<LoginScreen>:
    name: 'login'
    BoxLayout:
        orientation: 'vertical'
        Label: 
            text: 'DMS system'
        TextInput:
            id: user
            hint_text: 'Username'
            multiline: False

        TextInput:
            id: passw
            hint_text: 'Password'
            multiline: False
            password: True
        Button:
            text: 'Login'
            on_press:
                app.login(user.text, passw.text)

        Label:
            id: stat
            text: ''

<MainScreen>:
    name: 'main'

    

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
            

        BoxLayout:
            Label:
                text: app.d3
            Label:
                text: app.finl


        TextInput:
            id: url
            hint_text: 'enter url'
            multiline: False


        BoxLayout:
        Label: 
            text: "drowsiness values"
        Slider:
            id: slider1
            min: 2
            max: 12
            value: app.drowsthresh
            step: 1
            orientation: 'horizontal'
            on_value: app.drowsthresh = int(self.value)
            
        Label: 
            text: "distraction values"
        Slider:
            id: slider2
            min: 2
            max: 12
            value: app.distthresh
            step: 1
            orientation: 'horizontal'
            on_value: app.distthresh = int(self.value)
                
        
        BoxLayout:
            size_hint_y: None
            height: '48dp'
            padding: '4dp'

            ToggleButton:
                text: 'Start' if self.state == 'normal' else 'Stop'
                on_state:
                    app.start(1000, 0, url.text) if self.state == 'down' else \
                    app.stop()
            Button:
                text: 'monitor'
                on_press: app.dd(url.text)

<RegisterScreen>:
    name: 'register'
    BoxLayout:
        orientation: 'vertical'
        Label: 
            text: 'New User Registration'
        TextInput:
            id: user
            hint_text: 'Username'
            multiline: False
        TextInput:
            id: passw
            hint_text: 'Password'
            multiline: False
            password: True
        TextInput:
            id: fname
            hint_text: 'First Name'
            multiline: False
        TextInput:
            id: lname
            hint_text: 'Last Name'
            multiline: False
        TextInput:
            id: brand
            hint_text: 'Car Brand'
            multiline: False
        TextInput:
            id: model
            hint_text: 'Car Model'
            multiline: False
        Button:
            text: 'Register'
            on_press: app.register(user.text, passw.text, fname.text, lname.text, brand.text, model.text)
        Button:
            text: 'Back to Login'
            on_press: app.root.current = 'login'
        Label:
            id: stat
            text: ''
    
'''
class LoginScreen(Screen): pass
class MainScreen(Screen): pass
class RegisterScreen(Screen): pass


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

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image from the url, convert it to a NumPy array, and then read it into OpenCV format
    resp = urlopen(url)
    if resp is not None:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, readFlag)
        image = cv2.resize(image, (256,256))
        return image

def detect_and_crop_face(frame):
    """ Detects face in frame and crops. Also returns ]coordinates for bounding box. If no face, returns None."""
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
            "role": "user","parts": [{"text": user_message}]})

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
    gps_location = StringProperty()
    gps_status = StringProperty('Click Start to get GPS location updates')
    drows = StringProperty("drows")
    time = StringProperty("latency")
    finl = StringProperty("final status")


    bot = GeminiChatbot("")
    driver_full_name = None
    msg= False
    counter = 0  #counts frames in focus
    drowsCounter = 0 #counts drowsy frames
    distcounter = 0 #counts distracted frames

    drowsthresh = 8
    distthresh = 8


    #for testing purposes, delete in final ver
    d1 =StringProperty()
    d2 =StringProperty()
    d3 =StringProperty()

    latitude = StringProperty()
    longitude = StringProperty()
    speed = StringProperty()
    web = StringProperty()

    def login(self, username, password):
        users_ref = db.collection("useracc")
        existing = users_ref.where("username", "==", username).get()
        if not existing:
            login = self.root.get_screen('login')
            login.ids.stat.text = "❌ User not found, please register first"
        else:
            if validate_user(username, password):
                full_name = get_driver_full_name(username)
                if full_name:
                    self.driver_full_name = full_name
                else:
                    self.driver_full_name = username.title()
                self.root.current = 'main'
                print(f"✅ Logged in as {self.driver_full_name}")
            else:
                login = self.root.get_screen('login')
                login.ids.stat.text = "❌ Wrong password"


    def register(self, username, password, fname, lname, brand, model):
        success = add_user_account(username, password, fname, lname, brand, model)
        if success:
            self.driver_full_name = f"{fname} {lname}".title()
            self.root.current = 'main'
        else:
            reg = self.root.get_screen('register')
            reg.ids.stat.text = "⚠️ User already exists"


    def build(self):
        try:
            gps.configure(on_location=self.on_location, on_status=self.on_status)
        except NotImplementedError:
            import traceback
            traceback.print_exc()
            self.gps_status = 'GPS is not implemented for your platform'

        return Builder.load_string(kv)

#functions start to on_resume are for gps
    def start(self, minTime, minDistance, text):
        gps.start(minTime, minDistance)
        tts.speak(self.bot.send_message("You are the **Driver Monitoring Assistant**. "
                    "Your job is to **detect drowsiness, check driver alertness, and ensure road safety**. "
                    "If the driver reports feeling tired, recommend stopping for rest or drinking coffee. "
                    "Ask how long they have been driving and give advice based on their response. "
                    "Keep responses short and direct but lighthearted or witty. dont use the same response more than once. "))
        Clock.unschedule(self.loadVid)
        Clock.schedule_interval(self.loadVid, 1/12)
        self.web = text

    def stop(self):
        gps.stop()

    def on_location(self, **kwargs):
        self.latitude = str(kwargs.get('lat'))
        self.longitude = str(kwargs.get('lon'))
        self.speed = str(kwargs.get('speed'))
        self.gps_location = "Latitude:" + (self.latitude)+"\nLongitude:" + (self.longitude)

    def on_status(self, stype, status):
        self.gps_status = 'type={}\n{}'.format(stype, status)


    def on_pause(self):
        gps.stop()
        return True

    def on_resume(self):
        gps.start(1000, 0)
        pass
    
    def dd(self, text):
        Clock.unschedule(self.loadVid)
        Clock.schedule_interval(self.loadVid, 1/12)
        self.web = text

    def loadVid(self, *args):
        
        try:
            frame = url_to_image("http://" +self.web+"/capture")         
        except  Exception as e:
            self.finl ="error cannot find camera"
            Clock.unschedule(self.loadVid)
            return 
        start = time.time()
        if frame is not None:
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

                if self.driver_full_name:
                    self.latitude = "16" 
                    self.longitude = "121"
                    log_to_firestore(self.driver_full_name, predicted_label, self.latitude, self.longitude)

                if predicted_label == "focused":
                    self.counter = self.counter +1           
                    if self.counter > 4:
                        self.finl ="FOCUSED"
                        self.distcounter = 0
                        self.drowsCounter = 0
                        self.counter =self.counter%20
                        self.msg = False
                
                #drowsy       
                elif (predicted_label == "drowsy" ) or (predicted_label == "yawning"):
                    self.drowsCounter = self.drowsCounter+1
                    self.counter = 0
                    if self.drowsCounter > self.drowsthresh:
                        self.finl ="SLEEPY"
                        if not self.msg:
                            tts.speak(self.bot.send_message("im sleepy"))
                            self.msg = True
                    self.drowsCounter =  self.drowsCounter%20

                #distracted
                else:
                    self.distcounter = self.distcounter +1
                    self.counter = 0
                    if self.distcounter >self.distthresh:
                        self.finl ="DISTRACTED"
                        if not self.msg:
                            tts.speak(self.bot.send_message("im using my phone"))
                            self.msg = True
                    self.distcounter = self.distcounter%20

        end = time.time()
        self.time = str(end - start)
        self.d1 = "counter: " +str(self.counter)  
        self.d2 = "drowscounter: "+str(self.drowsCounter)
        self.d3 = "distcounter: "+str(self.distcounter)
        
        self.gps_location = f'Latitude: {self.latitude}\nLongitude: {self.longitude}\nSpeed: {self.speed}'


if __name__ == '__main__':
    GpsTest().run()
