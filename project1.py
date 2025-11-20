import cv2
import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn as nn
import time
import math
import threading
import os
from ultralytics import YOLO
import mediapipe as mp
from collections import defaultdict

# ==============================
# CONFIGURATION
# ==============================

# Audio Settings
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0
MFCC_COUNT = 20
AUDIO_INPUT_DIM = 20
AUDIO_MODEL_PATH = "fight-detection/models/fight_cnn.pth"
ALARM_PATH = "long_alarm.wav"
AUDIO_CONFIDENCE_THRESHOLD = 0.60
AUDIO_COOLDOWN = 0.3

# Video Settings
MAX_PEOPLE = 3
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF_THRESH = 0.35
YOLO_IOU_THRESH = 0.5
CROP_PADDING = 0.1
MIN_BOX_AREA = 3000
POSE_FRAME_SKIP = 2

# Violence Thresholds
VISUAL_VIOLENCE_THRESHOLD = 750  # pixels per second
COMBINED_ALERT_THRESHOLD = 70     # Combined risk percentage

# Mediapipe Landmarks
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16

# Smoothing
SPEED_HISTORY_SIZE = 5

# ==============================
# GLOBAL STATE
# ==============================

# Audio state
last_audio_msg_time = 0
alarm_active = False
audio_violence_detected = False
audio_confidence = 0.0

# Video state
prev_wrist_positions = defaultdict(lambda: {"L": None, "R": None})
prev_times = defaultdict(lambda: None)
last_pose_landmarks = defaultdict(lambda: None)
violence_status = defaultdict(lambda: False)
max_speeds = defaultdict(lambda: 0)
speed_history = defaultdict(lambda: [])
frame_count = 0

# Combined threat level
combined_threat_level = 0.0

# ==============================
# AUDIO MODEL
# ==============================

class FightClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FightClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load audio model
audio_model = FightClassifier(AUDIO_INPUT_DIM)
try:
    audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location="cpu"))
    audio_model.eval()
    print("✅ Audio model loaded successfully")
except Exception as e:
    print(f"⚠️  Audio model not loaded: {e}")
    audio_model = None

# ==============================
# ALERT FUNCTIONS
# ==============================

def play_alarm():
    global alarm_active
    if alarm_active:
        return
    alarm_active = True
    try:
        os.system(f'afplay "{ALARM_PATH}"')  # macOS
    except:
        try:
            os.system(f'aplay "{ALARM_PATH}"')  # Linux
        except:
            print("🔊 ALARM!")  # Fallback
    alarm_active = False

def show_notification(message):
    try:
        # macOS
        os.system(f'''osascript -e 'display notification "{message}" with title "🚨 Violence Alert!"' ''')
    except:
        try:
            # Linux with notify-send
            os.system(f'notify-send "🚨 Violence Alert!" "{message}"')
        except:
            print(f"🚨 ALERT: {message}")

# ==============================
# AUDIO PROCESSING
# ==============================

def extract_audio_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_COUNT)
    return np.mean(mfcc, axis=1)

def audio_callback(indata, frames, time_info, status):
    global last_audio_msg_time, audio_violence_detected, audio_confidence, combined_threat_level
    
    if audio_model is None:
        return
    
    audio = indata.flatten()
    features = extract_audio_features(audio)

    if features.shape[0] != AUDIO_INPUT_DIM:
        return

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = audio_model(x)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        conf = conf.item()
        pred = pred.item()

    now = time.time()
    if now - last_audio_msg_time < AUDIO_COOLDOWN:
        return

    last_audio_msg_time = now
    audio_confidence = conf

    if pred == 1 and conf >= AUDIO_CONFIDENCE_THRESHOLD:
        audio_violence_detected = True
        # Update combined threat
        combined_threat_level = max(combined_threat_level, conf * 100)
    else:
        audio_violence_detected = False

# ==============================
# VIDEO PROCESSING HELPERS
# ==============================

def expand_box(x1, y1, x2, y2, pad_frac, img_w, img_h):
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * pad_frac), int(h * pad_frac)
    return max(0, x1 - pad_w), max(0, y1 - pad_h), min(img_w, x2 + pad_w), min(img_h, y2 + pad_h)

def draw_point(img, x, y, label=None, radius=4, color=(0,255,0)):
    cv2.circle(img, (int(x), int(y)), radius, color, -1)
    if label:
        cv2.putText(img, label, (int(x)+5, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def normalized_to_pixel(lm, crop_box):
    x1, y1, x2, y2 = crop_box
    w, h = x2 - x1, y2 - y1
    return x1 + np.clip(lm.x, 0, 1) * w, y1 + np.clip(lm.y, 0, 1) * h

def compute_speed(prev_point, curr_point, dt):
    if prev_point is None or dt <= 0:
        return 0
    dx, dy = curr_point[0] - prev_point[0], curr_point[1] - prev_point[1]
    return math.sqrt(dx*dx + dy*dy) / dt

def smooth_speed(person_id, speed):
    speed_history[person_id].append(speed)
    if len(speed_history[person_id]) > SPEED_HISTORY_SIZE:
        speed_history[person_id].pop(0)
    return sum(speed_history[person_id]) / len(speed_history[person_id])

def draw_risk_bar(img, x, y, width, height, risk_level, label="RISK"):
    cv2.rectangle(img, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (180, 180, 180), 1)
    
    fill_width = int((risk_level / 100) * width)
    
    if risk_level < 30:
        color = (0, 255, 0)
    elif risk_level < 60:
        color = (0, 255, 255)
    elif risk_level < 85:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)
    
    if fill_width > 0:
        cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)
    
    text = f"{label}: {risk_level:.0f}%"
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

def draw_combined_threat_panel(img, people_risks, audio_threat, combined_threat):
    """Draw comprehensive threat assessment panel"""
    h, w = img.shape[:2]
    
    meter_width = 280
    meter_height = 30
    
    num_people = len(people_risks)
    panel_height = 140 + num_people * 35
    panel_width = meter_width + 30
    
    panel_x = w - panel_width - 20
    panel_y = h - panel_height - 20
    
    # Semi-transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                  (20, 20, 20), -1)
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                  (200, 200, 200), 2)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    meter_x = panel_x + 15
    meter_y = panel_y + 45
    
    # Title
    cv2.putText(img, "INTEGRATED THREAT ASSESSMENT", (meter_x - 5, panel_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    
    # Combined threat meter (most prominent)
    draw_risk_bar(img, meter_x, meter_y, meter_width, meter_height, combined_threat, "COMBINED")
    
    # Audio threat
    audio_color = (0, 0, 255) if audio_threat >= AUDIO_CONFIDENCE_THRESHOLD * 100 else (100, 100, 100)
    draw_risk_bar(img, meter_x, meter_y + 40, meter_width, 25, audio_threat, "AUDIO")
    
    # Visual threat (max of all people)
    visual_threat = max(people_risks.values()) if people_risks else 0
    draw_risk_bar(img, meter_x, meter_y + 75, meter_width, 25, visual_threat, "VISUAL")
    
    # Individual person risks
    y_offset = meter_y + 110
    for person_id in sorted(people_risks.keys()):
        risk = people_risks[person_id]
        draw_risk_bar(img, meter_x, y_offset, meter_width, 22, risk, f"P{person_id + 1}")
        y_offset += 35

# ==============================
# INITIALIZE MODELS
# ==============================

print("🚀 Initializing Integrated Violence Detection System...")
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_NAME)
yolo_model.fuse()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✅ Visual models loaded")

# ==============================
# MAIN EXECUTION
# ==============================

def main():
    global frame_count, combined_threat_level
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    # Start audio stream
    audio_stream = None
    if audio_model is not None:
        try:
            audio_stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                callback=audio_callback,
                blocksize=int(SAMPLE_RATE * AUDIO_DURATION)
            )
            audio_stream.start()
            print("✅ Audio monitoring started")
        except Exception as e:
            print(f"⚠️  Audio stream failed: {e}")
    
    prev_frame_time = time.time()
    last_alert_time = 0
    ALERT_COOLDOWN = 5  # seconds
    
    print("\n🔴 INTEGRATED VIOLENCE DETECTION SYSTEM ACTIVE")
    print("   📹 Visual Detection: ON")
    print("   🎤 Audio Detection:", "ON" if audio_model else "OFF")
    print("   Press 'q' or ESC to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            h, w = frame.shape[:2]
            
            # YOLO person detection
            results = yolo_model.predict(
                frame, 
                conf=YOLO_CONF_THRESH, 
                iou=YOLO_IOU_THRESH, 
                verbose=False,
                device='cpu',
                half=False
            )
            
            detections = []
            r = results[0]
            
            if hasattr(r.boxes, "xyxy") and len(r.boxes.xyxy) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confs, clss):
                    if cls == 0:  # person class
                        x1, y1, x2, y2 = box.astype(int)
                        if (x2 - x1) * (y2 - y1) > MIN_BOX_AREA:
                            detections.append((x1, y1, x2, y2, float(conf)))
            
            detections = sorted(detections, key=lambda x: x[4], reverse=True)[:MAX_PEOPLE]
            
            run_pose = (frame_count % POSE_FRAME_SKIP == 0)
            people_risks = {}
            
            # Process each detected person
            for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
                person_id = idx
                cx1, cy1, cx2, cy2 = expand_box(x1, y1, x2, y2, CROP_PADDING, w, h)
                
                box_color = (0, 0, 255) if violence_status[person_id] else (255, 0, 0)
                box_thickness = 3 if violence_status[person_id] else 2
                
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), box_color, box_thickness)
                
                label = f"P{idx+1} {conf:.2f}"
                if violence_status[person_id]:
                    label += " [VIOLENT]"
                cv2.putText(frame, label, (cx1, cy1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Pose estimation
                if run_pose:
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        mp_result = pose.process(crop_rgb)
                        
                        if mp_result.pose_landmarks:
                            last_pose_landmarks[person_id] = (mp_result.pose_landmarks, (cx1, cy1, cx2, cy2))
                
                # Use cached landmarks
                if last_pose_landmarks[person_id] is not None:
                    lm_data, crop_box = last_pose_landmarks[person_id]
                    lm = lm_data.landmark
                    
                    head_x, head_y = normalized_to_pixel(lm[NOSE], crop_box)
                    lsx, lsy = normalized_to_pixel(lm[LEFT_SHOULDER], crop_box)
                    rsx, rsy = normalized_to_pixel(lm[RIGHT_SHOULDER], crop_box)
                    torso_x, torso_y = (lsx + rsx) / 2, (lsy + rsy) / 2
                    lwx, lwy = normalized_to_pixel(lm[LEFT_WRIST], crop_box)
                    rwx, rwy = normalized_to_pixel(lm[RIGHT_WRIST], crop_box)
                    
                    # Draw skeleton
                    draw_point(frame, head_x, head_y, f"H{idx+1}")
                    draw_point(frame, torso_x, torso_y, f"T{idx+1}", color=(0,200,255))
                    draw_point(frame, lwx, lwy, f"Lw{idx+1}", color=(0,255,255))
                    draw_point(frame, rwx, rwy, f"Rw{idx+1}", color=(0,255,255))
                    
                    cv2.line(frame, (int(torso_x), int(torso_y)), (int(head_x), int(head_y)), (0,255,0), 2)
                    cv2.line(frame, (int(torso_x), int(torso_y)), (int(lwx), int(lwy)), (255,0,255), 2)
                    cv2.line(frame, (int(torso_x), int(torso_y)), (int(rwx), int(rwy)), (255,0,255), 2)
                    
                    # Violence detection
                    curr_time = time.time()
                    prev_time = prev_times[person_id]
                    dt = curr_time - prev_time if prev_time else 0
                    
                    curr_left, curr_right = (lwx, lwy), (rwx, rwy)
                    prev_left = prev_wrist_positions[person_id]["L"]
                    prev_right = prev_wrist_positions[person_id]["R"]
                    
                    left_speed = compute_speed(prev_left, curr_left, dt)
                    right_speed = compute_speed(prev_right, curr_right, dt)
                    
                    max_speed = max(left_speed, right_speed)
                    smoothed_speed = smooth_speed(person_id, max_speed)
                    max_speeds[person_id] = smoothed_speed
                    
                    risk_level = min(100, (smoothed_speed / VISUAL_VIOLENCE_THRESHOLD) * 100)
                    people_risks[person_id] = risk_level
                    
                    if smoothed_speed > VISUAL_VIOLENCE_THRESHOLD:
                        violence_status[person_id] = True
                    else:
                        violence_status[person_id] = False
                    
                    prev_wrist_positions[person_id]["L"] = curr_left
                    prev_wrist_positions[person_id]["R"] = curr_right
                    prev_times[person_id] = curr_time
            
            # Calculate combined threat level
            visual_threat = max(people_risks.values()) if people_risks else 0
            audio_threat = audio_confidence * 100 if audio_violence_detected else 0
            
            # Weighted combination (60% visual, 40% audio)
            combined_threat_level = (visual_threat * 0.6) + (audio_threat * 0.4)
            
            # Trigger alert if threshold exceeded
            now = time.time()
            if combined_threat_level >= COMBINED_ALERT_THRESHOLD and (now - last_alert_time) > ALERT_COOLDOWN:
                last_alert_time = now
                alert_msg = f"Combined threat: {combined_threat_level:.0f}% | Visual: {visual_threat:.0f}% | Audio: {audio_threat:.0f}%"
                print(f"\n🚨 VIOLENCE ALERT! {alert_msg}\n")
                threading.Thread(target=lambda: show_notification(alert_msg)).start()
                if combined_threat_level >= 85:
                    threading.Thread(target=play_alarm).start()
            
            # Draw threat panel
            draw_combined_threat_panel(frame, people_risks, audio_threat, combined_threat_level)
            
            # Audio indicator
            audio_status = "🎤 AUDIO: "
            if audio_model is None:
                audio_status += "OFF"
                audio_color = (100, 100, 100)
            elif audio_violence_detected:
                audio_status += f"⚠️  FIGHT ({audio_confidence:.2f})"
                audio_color = (0, 0, 255)
            else:
                audio_status += f"✓ Normal ({audio_confidence:.2f})"
                audio_color = (0, 255, 0)
            
            cv2.putText(frame, audio_status, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)
            
            # FPS
            now = time.time()
            fps = 1 / (now - prev_frame_time + 1e-6)
            prev_frame_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            cv2.imshow("Integrated Violence Detection System", frame)
            
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break
    
    finally:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("\n✅ System shut down successfully")

if __name__ == "__main__":
    main()