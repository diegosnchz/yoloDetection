import cv2
import torch
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

# --- CONFIGURATION ---
MODEL_PATH = 'yolov5s.pt'  # Placeholder until best.pt is ready
CONF_THRESHOLD = 0.5
cyan = (255, 255, 0)
magenta = (255, 0, 255)
red = (0, 0, 255)
green = (0, 255, 0)

# --- LOAD MODELS ---
print("[INFO] Loading YOLOv5 model...")
# Load local custom model if exists, else load generic yolov5s from hub
if Path('best.pt').exists():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    print("Loaded custom best.pt")
else:
    print("Warning: best.pt not found. Using standard yolov5s for demo.")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- UTILS ---
def is_inside(point, box):
    """Check if point (x,y) is inside box (x1,y1,x2,y2)."""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2

def draw_futuristic_hud(img, box, label, conf, color=cyan, active=False):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    # Base Box
    thickness = 2 if not active else 4
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Corner brackets (Iron Man style)
    l = 15 # length of bracket
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness)

    # Label with background
    text = f"{label} {conf:.2f}"
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    if active:
        # Floating stats
        cv2.line(img, (x2, y1), (x2 + 30, y1 - 30), color, 2)
        cv2.circle(img, (x2 + 30, y1 - 30), 3, color, -1)
        cv2.putText(img, "ANALYSIS: VERIFIED", (x2 + 35, y1 - 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        cv2.putText(img, f"CLASS: {label.upper()}", (x2 + 35, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

print("[INFO] Starting Sentinel HUD...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # 1. Hand Detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)
    
    index_finger_tip = None
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get Index Finger Tip (ID 8)
            x_tip = int(hand_landmarks.landmark[8].x * w)
            y_tip = int(hand_landmarks.landmark[8].y * h)
            index_finger_tip = (x_tip, y_tip)
            
            # Draw cursor
            cv2.circle(frame, index_finger_tip, 8, magenta, -1)
            cv2.circle(frame, index_finger_tip, 12, cyan, 1)

    # 2. YOLO Detection
    results_yolo = model(frame_rgb)
    detections = results_yolo.xyxy[0] # x1, y1, x2, y2, conf, cls
    
    alert_triggered = False
    
    for *box, conf, cls_id in detections:
        label = results_yolo.names[int(cls_id)]
        
        # Color mapping
        color = cyan
        if "Sin_Equipo" in label or "Peligro" in label:
            color = red
            alert_triggered = True
            
        # Check interaction
        is_active = False
        if index_finger_tip:
             if is_inside(index_finger_tip, box):
                 is_active = True
        
        draw_futuristic_hud(frame, box, label, conf, color, active=is_active)

    # 3. Security Alert Effect
    if alert_triggered:
        # Red overlay flash
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), red, -1)
        alpha = 0.2 + 0.1 * np.sin(time.time() * 10) # Pulping effect
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "WARNING: SAFETY BREACH", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, red, 2)

    cv2.imshow('SENTINEL HUD', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
