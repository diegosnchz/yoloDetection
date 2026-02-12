import streamlit as st
import cv2
import torch
import numpy as np
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# --- MEDIAPIPE IMPORT FIX ---
import mediapipe as mp
try:
    # Most robust way to access solutions
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except (AttributeError, ImportError):
    try:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_drawing
    except ImportError:
        mp_hands = None
        mp_drawing = None

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SENTINEL HUD - Core Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00ffcc;
        text-align: center;
    }
    .stAlert {
        background-color: #3d0000;
        border: 1px solid #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_yolo():
    model_path = Path('best.pt')
    if model_path.exists():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

@st.cache_resource
def load_hands():
    if mp_hands:
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    return None

yolo_model = load_yolo()
hands_model = load_hands()

# --- WEBRTC CONFIG ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class SentinelTransformer(VideoTransformerBase):
    def __init__(self):
        self.conf_threshold = 0.45
        self.alert_mode = True

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. YOLO Detection
        results = yolo_model(img)
        detections = results.pandas().xyxy[0]
        detections = detections[detections['confidence'] >= self.conf_threshold]
        
        # 2. Hand Tracking
        if hands_model:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hands_model.process(img_rgb)
            # Gesture logic could be expanded here
            
        # 3. HUD Rendering
        alert_count = 0
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            label = det['name']
            conf = det['confidence']
            
            color = (0, 255, 204) # Neo Green
            if "Sin_Equipo" in label or "Danger" in label:
                color = (0, 0, 255) # Red
                alert_count += 1
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return img

# --- SIDEBAR ---
st.sidebar.title("🛡️ SENTINEL CORE")
st.sidebar.info("Industrial Security Interface v3.0 (WebRTC)")

if not hands_model:
    st.sidebar.warning("⚠️ Hand Tracking Module Error (Host Library Incompatibility)")

conf_slider = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
alert_toggle = st.sidebar.toggle("Security Lockdown Mode", value=True)

# --- HEADER ---
st.title("SENTINEL HUD: Industrial Command Dashboard")
st.write("Real-time PPE Detection via Browser Stream")

# --- MAIN DASHBOARD ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Operations - Secure Stream")
    webrtc_ctx = webrtc_streamer(
        key="sentinel-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=SentinelTransformer,
        async_transform=True,
    )
    
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.conf_threshold = conf_slider
        webrtc_ctx.video_transformer.alert_mode = alert_toggle

with col2:
    st.subheader("System Metrics")
    m1, m2 = st.columns(2)
    m1.metric("System Status", "ONLINE", delta="ACTIVE")
    m2.metric("Environment", "SECURE")
    
    st.subheader("Instructions")
    st.markdown("""
    1. **Allow Camera Access** when prompted by the browser.
    2. **Press START** to begin the encrypted stream.
    3. The system will detect PPE and Security breaches automatically.
    """)
    
    st.subheader("Diagnostic Info")
    st.write(f"Python Version: {np.__version__} (via Numpy)")
    if hands_model:
        st.success("Mediapipe Core: ATTACHED")
    else:
        st.error("Mediapipe Core: DETACHED (Check Logs)")

# --- FOOTER ---
if st.button("Reset Dashboard Statistics"):
    st.session_state.alerts = []
    st.rerun()
