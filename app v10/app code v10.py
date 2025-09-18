from kivy.lang import Builder
from plyer import gps
from kivy.app import App
from kivy.properties import StringProperty
from kivy.clock import Clock, mainthread
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.slider import Slider
from plyer import tts
import math, time
import cv2
import torch
import numpy as np
import requests
from torchvision import transforms
from PIL import Image as im
import random
#import time
from kivy.core.audio import SoundLoader
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

# ── MAIN CODES ──────────────────────────────────
class LoginScreen(Screen): pass
class MainScreen(Screen): pass
class RegisterScreen(Screen): pass
#-------GEMINI CODES -----------------------------------------
class GeminiChatbot:
    def __init__(self, api_key, model="gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        self.headers = {
            "Content-Type": "application/json"}
        self.chat_history = []  # Proper Gemini format: list of dicts with 'role' and 'parts'

    def send_message(self, user_message):
        # Add the user message to history
        self.chat_history.append({
            "role": "user","parts": [{"text": user_message}]})

        data = { "contents": self.chat_history}
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
        
#--------------------------------------------------------------
#HELPER CODES
# ===================== Config: thresholds & timing ============================
# Angles in degrees (relative to your zero). Negative pitch = looking down.
YAW_WARN_DEG   = 35.0
YAW_ALERT_DEG  = 45.0
PITCHDN_WARN   = -15.0
PITCHDN_ALERT  = -25.0



# Which classifier labels count as warn/alert
CLASS_ALERT = {"drowsy", "yawning","using phone"}  # removed WARN
# Smoothing for pose (0..1): higher = snappier; lower = smoother
SMOOTH_ALPHA   = 0.25
# Beep cooldown so we don't spam beeps
BEEP_COOLDOWN_S = 3.0
# If yaw feels inverted on your setup, flip this:
INVERT_YAW = False

#====================== for url to image====================
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image from the url, convert it to a NumPy array, and then read it into OpenCV format
    resp = urlopen(url)
    if resp is not None:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, readFlag)
        image = cv2.resize(image, (256,256))
        return image


# ===================== YuNet model (face detector / 5 landmarks) =============
FD_MODEL_NAME = "face_detection_yunet_2023mar.onnx"

def create_yunet(model_path: str, size=(320, 320)):
    # OpenCV exposes either FaceDetectorYN_create(...) or FaceDetectorYN.create(...)
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(model_path, "", size, 0.6, 0.3, 5000)
    if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
        return cv2.FaceDetectorYN.create(model_path, "", size, 0.6, 0.3, 5000)
    #raise SystemExit("YuNet not found. Install opencv-contrib-python.")

# ===================== Head-pose utils =======================================
def rotmat_to_euler_deg(R: np.ndarray):
    """Return (pitch, yaw, roll) in degrees. x-right, y-down, z-forward."""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(R[2,1], R[2,2]))
        yaw   = math.degrees(math.atan2(-R[2,0], sy))
        roll  = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        pitch = math.degrees(math.atan2(-R[1,2], R[1,1]))
        yaw   = math.degrees(math.atan2(-R[2,0], sy))
        roll  = 0.0
    return np.array([pitch, yaw, roll], dtype=float)

def ema(prev, new, alpha=0.25):
    return new if prev is None else alpha*new + (1-alpha)*prev

def shortbeep():
    print("WARNING")

#be sure to change this
def beep():
    sound = SoundLoader.load('alarm.wav')
    sound.play()
  

# ===================== Classifier setup ======================================
# Put your candidate model filenames here (TorchScript .pt preferred)

# Default label set (will auto-expand if mismatch with model outputs)
CLASS_NAMES = ['drowsy', 'focused', 'using phone', 'yawning']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf = torch.jit.load("DMS_mobilevitv2_100.pt", map_location=device)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])





class GpsTest(App):
    gps_location = StringProperty()
    gps_status = StringProperty('Click Start to get GPS location updates')
    drows = StringProperty("drows")
    timqe = StringProperty("latency")
    finl = StringProperty("final status")
    d1 =StringProperty()
    d2 =StringProperty()
    d3 =StringProperty()
    latitude = StringProperty()
    longitude = StringProperty()
    speed = StringProperty()
    web = StringProperty()

    bot = GeminiChatbot("")
    driver_full_name = None
    msg= False
    
    warn = StringProperty()
    alert = StringProperty()
    noface = StringProperty()


    # Dwell times (seconds)
    DWELL_WARN_S   = 1.0                     #put these 3 as variables instead,  make a 3rd slider
    DWELL_ALERT_S  = 2.0
    NOFACE_ALERT_S = 1.0
    # Classifier dwell (use same by default; tweak if you like)
    CLF_WARN_DWELL_S  = DWELL_WARN_S
    CLF_ALERT_DWELL_S = DWELL_ALERT_S

    # Probe output size once to align class names
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        out = clf(dummy.to(device))
        num_classes = int(out.shape[-1])
    global CLASS_NAMES
    if len(CLASS_NAMES) != num_classes:
        CLASS_NAMES = [f"class_{i}" for i in range(num_classes)]
        #print(f"[Classifier] Adjusted class_names to {num_classes} outputs:", CLASS_NAMES)

    # -------- YuNet face detector ----------
    yunet = create_yunet(str(FD_MODEL_NAME))


    # -------- 3D model (order must match YuNet's 5 landmarks) ----------
    MODEL_3D = np.array([
        ( 32.0,  38.0, -30.0),  # right eye (subject's RIGHT)
        (-32.0,  38.0, -30.0),  # left eye
        (  0.0,   0.0,   0.0),  # nose tip
        ( 28.0, -28.0, -30.0),  # right mouth
        (-28.0, -28.0, -30.0),  # left mouth
    ], dtype=np.float32)
    axis = np.float32([[0,0,0],[80,0,0],[0,80,0],[0,0,120]])
    smoothed = None
    zero = np.zeros(3, dtype=float)

    # -------- Timers & state ----------
    last_seen_face = time.monotonic()
    yaw_warn_start = yaw_alert_start = None
    pitch_warn_start = pitch_alert_start = None
    clf_warn_start = clf_alert_start = None
    last_beep = 0.0
    last_msg = 0.0
    state = "OK"  # OK / WARN / ALERT
    last_pred_label, last_pred_conf = None, 0.0











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

        return Builder.load_file("layout.kv")

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
        Clock.unschedule(self.loadVid)

    def on_location(self, **kwargs):
        self.latitude = str(kwargs.get('lat'))
        self.longitude = str(kwargs.get('lon'))
        self.speed = str(kwargs.get('speed'))

    def on_status(self, stype, status): self.gps_status = 'type={}\n{}'.format(stype, status)

    def on_pause(self):
        gps.stop()
        return True

    def on_resume(self):
        gps.start(1000, 0)
        pass
    
    def dd(self, text):
        print(self.bot.send_message("You are the **Driver Monitoring Assistant**. "
                    "Your job is to **detect drowsiness, check driver alertness, and ensure road safety**. "
                    "If the driver reports feeling tired, recommend stopping for rest or drinking coffee. "
                    "Ask how long they have been driving and give advice based on their response. "
                    "Keep responses short and direct but lighthearted or witty. dont use the same response more than once. "))
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

        h, w = frame.shape[:2]
        self.yunet.setInputSize((w, h))
        det_out = self.yunet.detect(frame)
        faces = det_out[1] if isinstance(det_out, tuple) and len(det_out) == 2 else det_out

        now = time.monotonic()
        reasons = []
        bbox_drawn = False


        if faces is not None and len(faces) > 0:
            self.last_seen_face = now
            idx = int(np.argmax(faces[:, 14]))  # highest confidence
            best = faces[idx]

            # YuNet layout: [x,y,w,h, rx,ry, lx,ly, nx,ny, rmx,rmy, lmx,lmy, score]
            x, y, bw, bh = best[:4].astype(int)
            x = max(0, x); y = max(0, y)
            bw = min(bw, w - x); bh = min(bh, h - y)
            bbox_drawn = True

            # 5 landmarks for pose
            pts2d = np.array([
                best[4:6], best[6:8], best[8:10], best[10:12], best[12:14]
            ], dtype=np.float32).reshape(-1, 2)

            # Camera intrinsics
            cam_mtx = np.array([[w, 0, w/2],
                                [0, w, h/2],
                                [0, 0,   1 ]], dtype=np.float32)
            dist = np.zeros((4,1), dtype=np.float32)

            # Pose (EPNP -> optional refine)
            ok_pnp, rvec, tvec = cv2.solvePnP(self.MODEL_3D, pts2d, cam_mtx, dist, flags=cv2.SOLVEPNP_EPNP)
            if ok_pnp and hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(self.MODEL_3D, pts2d, cam_mtx, dist, rvec, tvec)

            if ok_pnp:
                R, _ = cv2.Rodrigues(rvec)
                angles = rotmat_to_euler_deg(R)
                if INVERT_YAW:
                    angles[1] = -angles[1]
                self.smoothed = ema(self.smoothed, angles, alpha=SMOOTH_ALPHA)
                disp = self.smoothed - self.zero
                pitch, yaw, roll = float(disp[0]), float(disp[1]), float(disp[2])

                # Pose thresholds -> timers
                if abs(yaw) >= YAW_ALERT_DEG:  self.yaw_alert_start = self.yaw_alert_start or now
                else:                           self.yaw_alert_start = None
                if abs(yaw) >= YAW_WARN_DEG:   self.yaw_warn_start  = self.yaw_warn_start  or now
                else:                           self.yaw_warn_start  = None

                if pitch <= PITCHDN_ALERT:     self.pitch_alert_start = self.pitch_alert_start or now
                else:                           self.pitch_alert_start = None
                if pitch <= PITCHDN_WARN:      self.pitch_warn_start  = self.pitch_warn_start  or now
                else:                           self.pitch_warn_start  = None

                # Draw landmarks & axes
                for (px, py) in pts2d.astype(int):
                    cv2.circle(frame, (px, py), 3, (0, 200, 255), -1)
                imgpts, _ = cv2.projectPoints(self.axis, rvec, tvec, cam_mtx, dist)
                p0 = tuple(np.int32(imgpts[0].ravel()))
                cv2.line(frame, p0, tuple(np.int32(imgpts[1].ravel())), (255,0,0), 2)
                cv2.line(frame, p0, tuple(np.int32(imgpts[2].ravel())), (0,255,0), 2)
                cv2.line(frame, p0, tuple(np.int32(imgpts[3].ravel())), (0,0,255), 2)

                # Small HUD (top-left)                
                #print(f"Pitch: {pitch:6.1f}" + f"Yaw:   {yaw:6.1f}"+f"Roll:  {roll:6.1f}")
                self.d1 = f"Pitch: {pitch:6.1f}"
                self.d2 = f"Yaw:   {yaw:6.1f}"
                self.d3 = f"Roll:  {roll:6.1f}"

            # Classifier: crop face from YuNet bbox (BGR->RGB->PIL->tensor)
            face_crop = frame[y:y+bh, x-15:x+bw+15]
            if face_crop.size != 0:
                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_image = im.fromarray(rgb_face)
                input_tensor = TRANSFORM(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = clf(input_tensor)[0]
                    probs = torch.softmax(outputs, dim=0)
                    conf, pred_idx = torch.max(probs, dim=0)
                last_pred_label = CLASS_NAMES[int(pred_idx)]
                last_pred_conf  = float(conf)
                print(last_pred_label)
                self.drows = last_pred_label
                # Class dwell timers
                if last_pred_label in CLASS_ALERT:
                    self.clf_alert_start = self.clf_alert_start or now
                else:
                    self.clf_alert_start = None

                if  (last_pred_label in CLASS_ALERT):
                    self.clf_warn_start = self.clf_warn_start or now
                else:
                    self.clf_warn_start = None

                # Draw bbox + label
                cv2.rectangle(frame, (x-15, y), (x+bw+15, y+bh), (255, 255, 255), 2)
                cv2.putText(frame, f'{last_pred_label} ({last_pred_conf:.2f})',
                            (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        # ====== State machine (pose + classifier + no-face) ======
        noface_dur = now - self.last_seen_face
        noface_alert = noface_dur >= self.NOFACE_ALERT_S

        yaw_warn_dur   = (now - self.yaw_warn_start)    if self.yaw_warn_start    else 0.0
        yaw_alert_dur  = (now - self.yaw_alert_start)   if self.yaw_alert_start   else 0.0
        pit_warn_dur   = (now - self.pitch_warn_start)  if self.pitch_warn_start  else 0.0
        pit_alert_dur  = (now - self.pitch_alert_start) if self.pitch_alert_start else 0.0

        clf_warn_dur   = (now - self.clf_warn_start)    if self.clf_warn_start    else 0.0
        clf_alert_dur  = (now - self.clf_alert_start)   if self.clf_alert_start   else 0.0

        new_state = "OK"
        # Any alert reason?
        if (noface_alert or
            yaw_alert_dur  >= self.DWELL_ALERT_S or
            pit_alert_dur  >= self.DWELL_ALERT_S or
            clf_alert_dur  >= self.DWELL_ALERT_S):
            new_state = "ALERT"
        # Else, any warn reason?
        elif (yaw_warn_dur >= self.DWELL_WARN_S or
              pit_warn_dur >= self.DWELL_WARN_S or
              clf_warn_dur >= self.DWELL_WARN_S):
            new_state = "WARN"
        
#=====================================================================
        # # Transition actions
        if new_state != self.state:
            self.state = new_state
            if self.state == "ALERT" and (now - self.last_msg) >= BEEP_COOLDOWN_S:
                print("ALERT")
                #tts.speak(self.bot.send_message("im using my phone"))
                print(self.bot.send_message("im not paying attention to the road"))
                self.last_msg = now
            if self.state == "WARN" and (now - self.last_beep) >= BEEP_COOLDOWN_S:
                print("WARN")
                beep()
                
        self.finl = self.state

        if self.driver_full_name:
            self.latitude = "16" 
            self.longitude = "121"
            log_to_firestore(self.driver_full_name, self.state, self.latitude, self.longitude)
        end = time.time()
        self.timqe = str(end - start)
        print(str(self.DWELL_WARN_S))
        self.warn = str(self.DWELL_WARN_S)
        self.alert = str(self.DWELL_ALERT_S)
        self.noface = str(self.NOFACE_ALERT_S)

if __name__ == '__main__':
    GpsTest().run()
