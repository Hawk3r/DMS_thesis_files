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

kv = '''
BoxLayout:
    orientation: 'vertical'

    Label:
        text: app.gps_location

    Label:
        text: app.gps_status
    
    Label:
        text: app.drows

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


def detect_and_crop_face(frame):
    """ Detects face in frame and crops. Also returns ]coordinates for bounding box. If no face, returns None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face_crop = frame[y:y+h, x:x+w]  # Crop face region (remains in RGB)
        return face_crop, (x, y, w, h)
    
    return None, None




class GpsTest(App):
    capture = cv2.VideoCapture()
    gps_location = StringProperty()
    gps_status = StringProperty('Click Start to get GPS location updates')
    drows = StringProperty("drows")
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
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.loadVid , 1)

    
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


if __name__ == '__main__':
    GpsTest().run()
