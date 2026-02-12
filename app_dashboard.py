import streamlit as st
import cv2
import torch
import numpy as np
import time
from pathlib import Path
import sqlite3
import tempfile
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# --- MEDIAPIPE IMPORT FIX ---
import mediapipe as mp
try:
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
    page_title="SENTINEL HUD // V3.1",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- FRONTEND DESIGN SKILL INJECTION: INDUSTRIAL CYBERPUNK ---
st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    /* GLOBAL THEME */
    body {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'Rajdhani', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at center, #0a0a12 0%, #000000 100%);
    }

    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.4);
    }
    
    .stText, p, label {
        font-family: 'Share Tech Mono', monospace;
        color: #a0a0a0;
    }

    /* HUD CONTAINERS */
    .hud-container {
        border: 1px solid #1a1a1a;
        background: rgba(20, 20, 30, 0.6);
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        border-left: 2px solid #00ffcc;
        margin-bottom: 20px;
        backdrop-filter: blur(5px);
    }

    /* METRICS */
    div[data-testid="stMetric"] {
        background-color: rgba(0, 255, 204, 0.05);
        border: 1px solid #00ffcc;
        border-radius: 0px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
        border-color: #fff;
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem !important;
        color: #00ffcc !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 2.5rem !important;
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }

    /* BUTTONS */
    div.stButton > button {
        background: transparent;
        border: 1px solid #00ffcc;
        color: #00ffcc;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-radius: 0px;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background: #00ffcc;
        color: #000;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.6);
    }

    /* ALERTS */
    .alert-box {
        background: rgba(255, 0, 50, 0.1);
        border: 1px solid #ff0033;
        color: #ff0033;
        padding: 10px;
        font-family: 'Share Tech Mono', monospace;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 50, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 50, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 50, 0); }
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #333;
    }

    </style>
    """, unsafe_allow_html=True)

DB_PATH = Path(tempfile.gettempdir()) / "sentinel_events.db"


def init_events_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_unix REAL NOT NULL,
            ts_iso TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def clear_events_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM events")
    conn.commit()
    conn.close()


def insert_event(label: str, confidence: float, severity: str, source: str = "webrtc"):
    now_unix = time.time()
    now_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_unix))
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(
        "INSERT INTO events (ts_unix, ts_iso, label, confidence, severity, source) VALUES (?, ?, ?, ?, ?, ?)",
        (now_unix, now_iso, label, float(confidence), severity, source),
    )
    conn.commit()
    conn.close()


def get_recent_events(limit: int = 30) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts_iso, severity, label, confidence, source FROM events ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,),
    )
    conn.close()
    return df


def get_event_metrics() -> dict:
    conn = sqlite3.connect(DB_PATH)
    now_unix = time.time()
    last_60s = now_unix - 60
    last_5m = now_unix - 300

    total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    high_last_60s = conn.execute(
        "SELECT COUNT(*) FROM events WHERE severity = 'ALTA' AND ts_unix >= ?",
        (last_60s,),
    ).fetchone()[0]
    high_last_5m = conn.execute(
        "SELECT COUNT(*) FROM events WHERE severity = 'ALTA' AND ts_unix >= ?",
        (last_5m,),
    ).fetchone()[0]
    medium_last_5m = conn.execute(
        "SELECT COUNT(*) FROM events WHERE severity = 'MEDIA' AND ts_unix >= ?",
        (last_5m,),
    ).fetchone()[0]

    sev_df = pd.read_sql_query(
        "SELECT severity, COUNT(*) AS count FROM events GROUP BY severity",
        conn,
    )
    conn.close()

    if high_last_5m > 0:
        threat = "HIGH"
    elif medium_last_5m > 2:
        threat = "MEDIUM"
    else:
        threat = "LOW"

    return {
        "total_events": total_events,
        "high_last_60s": high_last_60s,
        "high_last_5m": high_last_5m,
        "medium_last_5m": medium_last_5m,
        "threat": threat,
        "severity_counts": sev_df,
    }


def severity_for_label(label: str) -> str:
    label_l = label.lower()
    if "sin_equipo" in label_l or "danger" in label_l or "peligro" in label_l:
        return "ALTA"
    if "persona" in label_l:
        return "MEDIA"
    return "BAJA"

# --- LOAD MODELS ---
@st.cache_resource
def load_yolo():
    project_root = Path(__file__).resolve().parent
    model_path = project_root / 'best.pt'
    local_repo = project_root / "yolov5"
    if local_repo.exists():
        local_repo_path = str(local_repo.resolve())
        if model_path.exists():
            model = torch.hub.load(local_repo_path, 'custom', path=str(model_path), source='local')
        else:
            model = torch.hub.load(local_repo_path, 'yolov5s', source='local')
        return model

    if model_path.exists():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), trust_repo=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
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
        self.cooldown_seconds = 8
        self.last_alert_by_key = {}

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
            
        # 3. HUD Rendering - Cyberpunk Style
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            label = det['name']
            conf = det['confidence']
            severity = severity_for_label(label)
            
            color = (0, 255, 204) # Neo Green
            if "Sin_Equipo" in label or "Danger" in label:
                color = (0, 0, 255) # Neon Red
            
            # Corner Brackets only
            thickness = 2
            length = 20
            
            # Top Left
            cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
            cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
            # Top Right
            cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
            cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
            # Bottom Left
            cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
            cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
            # Bottom Right
            cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
            cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
            
            # Label Background
            (w, h), _ = cv2.getTextSize(f"{label} {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            if self.alert_mode and severity in {"ALTA", "MEDIA"}:
                now_t = time.time()
                alert_key = f"{severity}:{label}"
                last_t = self.last_alert_by_key.get(alert_key, 0.0)
                if now_t - last_t >= self.cooldown_seconds:
                    insert_event(label=label, confidence=float(conf), severity=severity, source="webrtc")
                    self.last_alert_by_key[alert_key] = now_t
            
        return img

init_events_db()

# --- SIDEBAR ---
st.sidebar.markdown("### SYSTEM CONFIG")

if not hands_model:
    st.sidebar.error("MODULE ERROR: HAND_TRACKING_404")

conf_slider = st.sidebar.slider("SENSITIVITY", 0.0, 1.0, 0.45)
alert_toggle = st.sidebar.toggle("LOCKDOWN PROTOCOL", value=True)
cooldown_slider = st.sidebar.slider("ALERT COOLDOWN (s)", 1, 60, 8)

if st.sidebar.button("PURGE EVENT HISTORY"):
    clear_events_db()
    st.sidebar.success("EVENT HISTORY PURGED")

st.sidebar.markdown("---")
st.sidebar.markdown("**STATUS**: OPERATIONAL")
st.sidebar.markdown("**VERSION**: 3.1.0-CYBER")

# --- HEADER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<h1>SENTINEL HUD <span style='font-size: 0.5em; color: #555;'>// CORE COMMAND</span></h1>", unsafe_allow_html=True)
with c2:
    st.markdown("<div style='text-align: right; color: #00ffcc; font-family: Share Tech Mono;'>SYS.TIME: " + time.strftime("%H:%M:%S") + "</div>", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='hud-container'><h3>LIVE UPLINK</h3>", unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="sentinel-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=SentinelTransformer,
        async_transform=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.conf_threshold = conf_slider
        webrtc_ctx.video_transformer.alert_mode = alert_toggle
        webrtc_ctx.video_transformer.cooldown_seconds = cooldown_slider

with col2:
    metrics = get_event_metrics()
    events_df = get_recent_events(limit=20)

    st.markdown("<div class='hud-container'><h3>TELEMETRY</h3>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("THREAT LEVEL", metrics["threat"])
    m2.metric("ACTIVE ALERTS (60s)", str(metrics["high_last_60s"]))
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("TOTAL EVENTS", str(metrics["total_events"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hud-container'><h3>SEVERITY MIX</h3>", unsafe_allow_html=True)
    if metrics["severity_counts"].empty:
        st.info("No events yet.")
    else:
        st.bar_chart(metrics["severity_counts"].set_index("severity"))
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='hud-container'><h3>EVENT LOG</h3>", unsafe_allow_html=True)
    if events_df.empty:
        st.markdown("<div style='font-family: Share Tech Mono; font-size: 0.8em; color: #888;'>[SYS] WAITING FOR EVENTS...</div>", unsafe_allow_html=True)
    else:
        display_df = events_df.copy()
        display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="EXPORT EVENTS CSV",
            data=csv_data,
            file_name="sentinel_events.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
