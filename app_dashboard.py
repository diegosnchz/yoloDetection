import hashlib
import sqlite3
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

st.set_page_config(
    page_title="SENTINEL Vision",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(1200px 500px at 50% -10%, #1a1a1a 0%, #0a0a0a 35%, #000 70%);
        color: #f5f5f5;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: #0b0b0b;
        border-right: 1px solid #222;
    }

    .title-wrap {
        padding: 8px 0 16px 0;
    }

    .title-main {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
    }

    .title-sub {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    .card {
        border: 1px solid #222;
        background: linear-gradient(180deg, rgba(18,18,18,0.95), rgba(10,10,10,0.95));
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }

    .card h3 {
        margin: 0;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #e5e7eb;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        background: #0f0f10;
        padding: 10px;
    }

    div.stButton > button {
        background: #f5f5f5;
        color: #0b0b0b;
        border: 1px solid #f5f5f5;
        border-radius: 10px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #ffffff;
        border-color: #ffffff;
    }

    .stDownloadButton > button {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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


def insert_event(label: str, confidence: float, severity: str, source: str = "upload"):
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


@st.cache_resource
def load_yolo():
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "best.pt"
    local_repo = project_root / "yolov5"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo custom: {model_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if local_repo.exists():
        local_repo_path = str(local_repo.resolve())
        model = torch.hub.load(local_repo_path, "custom", path=str(model_path), source="local")
    else:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(model_path), trust_repo=True)

    model.to(device)
    model.conf = 0.40
    model.eval()
    return model, device


def run_inference(image_rgb: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, pd.DataFrame]:
    model, _device = load_yolo()
    model.conf = conf_threshold
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model(image_bgr)
    detections = results.pandas().xyxy[0]
    detections = detections[detections["confidence"] >= conf_threshold].copy()

    annotated = image_bgr.copy()
    for _, det in detections.iterrows():
        x1, y1, x2, y2 = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
        label = str(det["name"])
        confidence = float(det["confidence"])

        severity = severity_for_label(label)
        color = (255, 255, 255)
        if severity == "ALTA":
            color = (0, 0, 255)
        elif severity == "MEDIA":
            color = (0, 200, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1 - 22), (x1 + 8 * len(text), y1), color, -1)
        cv2.putText(annotated, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated, detections


init_events_db()

if "last_alert_by_key" not in st.session_state:
    st.session_state.last_alert_by_key = {}
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_detections" not in st.session_state:
    st.session_state.last_detections = pd.DataFrame()
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None
if "last_conf_used" not in st.session_state:
    st.session_state.last_conf_used = None

st.sidebar.markdown("### Sentinel Controls")
conf_slider = st.sidebar.slider("Confidence", 0.0, 1.0, 0.45)
alert_toggle = st.sidebar.toggle("Enable alerts", value=True)
cooldown_slider = st.sidebar.slider("Cooldown (s)", 1, 60, 8)
if not (Path(__file__).resolve().parent / "best.pt").exists():
    st.sidebar.error("Falta best.pt en la raiz del proyecto. Copialo para usar el modelo custom de EPI.")
if st.sidebar.button("Purge event history"):
    clear_events_db()
    st.sidebar.success("History cleared")

st.markdown(
    """
    <div class="title-wrap">
        <p class="title-main">SENTINEL Vision</p>
        <p class="title-sub">Upload images, detect risk patterns, and track events in real time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="card"><h3>Image Inference</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    run_button = st.button("Run detection", use_container_width=True)

    if uploaded_file and run_button:
        image_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(image_bytes).hexdigest()
        same_input = (
            st.session_state.last_file_hash == file_hash
            and st.session_state.last_conf_used == conf_slider
            and st.session_state.last_result is not None
        )

        if not same_input:
            image_rgb = np.array(Image.open(uploaded_file).convert("RGB"))
            try:
                with st.spinner("Loading model (first time) and running inference..."):
                    annotated_bgr, detections_df = run_inference(image_rgb, conf_slider)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.info("Copia best.pt en la raiz del proyecto y vuelve a ejecutar.")
                st.stop()
            st.session_state.last_result = annotated_bgr
            st.session_state.last_detections = detections_df
            st.session_state.last_file_hash = file_hash
            st.session_state.last_conf_used = conf_slider
        else:
            detections_df = st.session_state.last_detections
            st.info("Using cached result for this image and confidence.")

        if alert_toggle and not detections_df.empty:
            for _, det in detections_df.iterrows():
                label = str(det["name"])
                confidence = float(det["confidence"])
                severity = severity_for_label(label)
                if severity in {"ALTA", "MEDIA"}:
                    key = f"{severity}:{label}:{file_hash}"
                    now_t = time.time()
                    last_t = st.session_state.last_alert_by_key.get(key, 0.0)
                    if now_t - last_t >= cooldown_slider:
                        insert_event(label=label, confidence=confidence, severity=severity, source="upload")
                        st.session_state.last_alert_by_key[key] = now_t

    if st.session_state.last_result is not None:
        st.image(st.session_state.last_result, channels="BGR", use_container_width=True)
        det_df = st.session_state.last_detections
        if det_df.empty:
            st.info("No detections above confidence threshold.")
        else:
            view_df = det_df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]].copy()
            view_df["confidence"] = (view_df["confidence"] * 100).round(2).astype(str) + "%"
            st.dataframe(view_df, use_container_width=True, hide_index=True)

with col_right:
    metrics = get_event_metrics()
    events_df = get_recent_events(limit=20)

    st.markdown('<div class="card"><h3>Telemetry</h3></div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Threat level", metrics["threat"])
    m2.metric("Active alerts (60s)", str(metrics["high_last_60s"]))
    st.metric("Total events", str(metrics["total_events"]))

    st.markdown('<div class="card"><h3>Severity mix</h3></div>', unsafe_allow_html=True)
    if metrics["severity_counts"].empty:
        st.info("No events yet.")
    else:
        st.bar_chart(metrics["severity_counts"].set_index("severity"))

    st.markdown('<div class="card"><h3>Event log</h3></div>', unsafe_allow_html=True)
    if events_df.empty:
        st.caption("Waiting for events...")
    else:
        display_df = events_df.copy()
        display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export events CSV",
            data=csv_data,
            file_name="sentinel_events.csv",
            mime="text/csv",
            use_container_width=True,
        )
