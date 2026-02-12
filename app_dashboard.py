import streamlit as st
import cv2
import torch
import mediapipe as mp
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
from PIL import Image

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
    }
    .alert-card {
        padding: 10px;
        background-color: #3d0000;
        border-left: 5px solid #ff0000;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # YOLO
    model_path = Path('best.pt')
    if model_path.exists():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    # MediaPipe
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    mp_drawing = mp.solutions.drawing_utils
    return model, hands, mp_drawing

model, hands, mp_drawing = load_models()

# --- STATE MANAGEMENT ---
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

# --- SIDEBAR ---
st.sidebar.title("🛡️ SENTINEL CORE")
st.sidebar.info("Industrial Security Interface v2.0")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
alert_mode = st.sidebar.toggle("Security Lockdown Mode", value=True)

# --- HEADER ---
st.title("SENTINEL HUD: Industrial Command Dashboard")
st.write("Real-time PPE Detection & Gesture Controlled Monitoring")

# --- DASHBOARD LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Operations Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("System Metrics")
    m1, m2 = st.columns(2)
    metric_detections = m1.metric("Live Detections", 0)
    metric_alerts = m2.metric("Active Alerts", 0)
    
    st.subheader("Classification Analytics")
    chart_placeholder = st.empty()
    
    st.subheader("Security Event Log")
    log_placeholder = st.empty()

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Wait: Camera not found or disconnected.")
            break
            
        # Processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.pandas().xyxy[0]
        
        # Filtering by confidence
        detections = detections[detections['confidence'] >= conf_threshold]
        
        # Hand tracking
        hand_results = hands.process(img_rgb)
        
        # HUD Logic & Drawing
        h, w, _ = frame.shape
        alert_count = 0
        current_detections = len(detections)
        
        # Draw Detections
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            label = det['name']
            conf = det['confidence']
            
            color = (0, 255, 204) # Neo Green
            if "Sin_Equipo" in label or "Danger" in label:
                color = (0, 0, 255) # Red
                alert_count += 1
                if alert_mode:
                    # Flash logic would go here, for now just a message
                    pass
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update Session State for Charts
        if alert_count > 0:
            st.session_state.alerts.append({
                "time": time.strftime("%H:%M:%S"),
                "event": "Security Breach detected",
                "severity": "High"
            })
            if len(st.session_state.alerts) > 10:
                st.session_state.alerts.pop(0)

        # Update Video
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # Update Metrics
        metric_detections.metric("Live Detections", current_detections)
        metric_alerts.metric("Active Alerts", alert_count, delta="ALERTA" if alert_count > 0 else None)
        
        # Update Chart
        if not detections.empty:
            counts = detections['name'].value_counts().reset_index()
            counts.columns = ['Object', 'Count']
            fig = px.bar(counts, x='Object', y='Count', color='Object', 
                         template="plotly_dark", color_discrete_sequence=["#00ffcc", "#ff0000", "#ffcc00"])
            fig.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
        # Update Log
        if st.session_state.alerts:
            df_log = pd.DataFrame(st.session_state.alerts[::-1])
            log_placeholder.table(df_log)
            
        time.sleep(0.01)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    st.error(f"Critical Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
