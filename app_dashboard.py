import hashlib
import os
import pathlib
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
DEFAULT_MODEL_NAME = "yolov5s"
DEFAULT_UI_CONFIDENCE = 0.30
AUTO_RETRY_MIN_CONFIDENCE = 0.25
AUTO_RETRY_FACTOR = 0.7


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
    if "sin_equipo" in label_l or "sin_epi" in label_l or "danger" in label_l or "peligro" in label_l:
        return "ALTA"
    if "persona" in label_l:
        return "MEDIA"
    return "BAJA"


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), float(width - 1)))
    y1 = max(0.0, min(float(y1), float(height - 1)))
    x2 = max(0.0, min(float(x2), float(width - 1)))
    y2 = max(0.0, min(float(y2), float(height - 1)))
    if x2 <= x1:
        x2 = min(float(width - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height - 1), y1 + 1.0)
    return x1, y1, x2, y2


def region_box_from_anchor(
    anchor: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    ax1, ay1, ax2, ay2 = anchor
    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    rx1, ry1, rx2, ry2 = region
    x1 = ax1 + aw * rx1
    y1 = ay1 + ah * ry1
    x2 = ax1 + aw * rx2
    y2 = ay1 + ah * ry2
    return clamp_box(x1, y1, x2, y2, width, height)


def refine_detections(detections: pd.DataFrame, image_shape: tuple[int, int, int]) -> pd.DataFrame:
    """Postprocess detections to avoid oversized false positives for PPE classes."""
    if detections.empty:
        return detections

    h, w = image_shape[:2]
    image_area = float(max(1, h * w))
    refined = detections.copy()

    label_series = refined["name"].astype(str).str.lower()
    person_mask = label_series.str.contains("persona")
    person_ref = None
    if person_mask.any():
        person_ref = refined.loc[person_mask].sort_values("confidence", ascending=False).iloc[0]
        person_ref_box = (
            float(person_ref["xmin"]),
            float(person_ref["ymin"]),
            float(person_ref["xmax"]),
            float(person_ref["ymax"]),
        )
    else:
        person_ref_box = None

    drop_rows: list[int] = []
    for idx, row in refined.iterrows():
        label_l = str(row["name"]).lower()
        x1 = float(row["xmin"])
        y1 = float(row["ymin"])
        x2 = float(row["xmax"])
        y2 = float(row["ymax"])
        box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area

        if "casco" in label_l and box_area_ratio > 0.22:
            anchor = person_ref_box if person_ref_box is not None else (x1, y1, x2, y2)
            nx1, ny1, nx2, ny2 = region_box_from_anchor(anchor, (0.30, 0.02, 0.72, 0.30), w, h)
            refined.at[idx, "xmin"] = nx1
            refined.at[idx, "ymin"] = ny1
            refined.at[idx, "xmax"] = nx2
            refined.at[idx, "ymax"] = ny2
            new_area_ratio = max(0.0, (nx2 - nx1) * (ny2 - ny1)) / image_area
            if new_area_ratio > 0.28:
                drop_rows.append(idx)

        if "chaleco" in label_l and box_area_ratio > 0.45:
            anchor = person_ref_box if person_ref_box is not None else (x1, y1, x2, y2)
            nx1, ny1, nx2, ny2 = region_box_from_anchor(anchor, (0.22, 0.25, 0.78, 0.80), w, h)
            refined.at[idx, "xmin"] = nx1
            refined.at[idx, "ymin"] = ny1
            refined.at[idx, "xmax"] = nx2
            refined.at[idx, "ymax"] = ny2

        # Remove detections that are still unrealistically large for helmet/vest.
        if "casco" in label_l:
            cx1 = float(refined.at[idx, "xmin"])
            cy1 = float(refined.at[idx, "ymin"])
            cx2 = float(refined.at[idx, "xmax"])
            cy2 = float(refined.at[idx, "ymax"])
            area_ratio = max(0.0, (cx2 - cx1) * (cy2 - cy1)) / image_area
            if area_ratio > 0.30:
                drop_rows.append(idx)

    if drop_rows:
        refined = refined.drop(index=drop_rows, errors="ignore")

    # Keep the strongest detection per class to reduce duplicate overlays.
    refined = refined.sort_values("confidence", ascending=False)
    refined = refined.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    return refined


def resolve_custom_model_path(project_root: Path) -> Path | None:
    """Find an available custom YOLO model in known locations."""
    candidates = [
        project_root / "best.pt",
        project_root / "yolov5" / "runs_academic" / "actividad1_50e" / "weights" / "best.pt",
        project_root / "yolov5" / "runs_academic" / "actividad1_10e" / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_cross_platform_checkpoint_loading() -> None:
    """Allow loading checkpoints saved on Windows when running in Linux containers."""
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath


def load_default_model(local_repo: Path):
    if local_repo.exists():
        return torch.hub.load(str(local_repo.resolve()), DEFAULT_MODEL_NAME, source="local")
    return torch.hub.load("ultralytics/yolov5", DEFAULT_MODEL_NAME, trust_repo=True)


@st.cache_resource
def load_yolo():
    project_root = Path(__file__).resolve().parent
    local_repo = project_root / "yolov5"
    model_path = resolve_custom_model_path(project_root)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if model_path is not None:
        try:
            ensure_cross_platform_checkpoint_loading()
            if local_repo.exists():
                local_repo_path = str(local_repo.resolve())
                model = torch.hub.load(local_repo_path, "custom", path=str(model_path), source="local")
            else:
                model = torch.hub.load("ultralytics/yolov5", "custom", path=str(model_path), trust_repo=True)
        except Exception as exc:
            st.warning(
                f"No se pudo cargar el modelo custom ({model_path.name}): {exc}. "
                f"Se usara {DEFAULT_MODEL_NAME}."
            )
            model = load_default_model(local_repo)
            model_path = None
    else:
        model = load_default_model(local_repo)

    model.to(device)
    model.conf = 0.40
    model.eval()
    return model, device, model_path


def run_inference(image_rgb: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, pd.DataFrame]:
    model, _device, _model_path = load_yolo()
    model.conf = conf_threshold
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model(image_bgr)
    detections = results.pandas().xyxy[0]
    detections = detections[detections["confidence"] >= conf_threshold].copy()
    detections = refine_detections(detections, image_bgr.shape)

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
if "last_effective_conf" not in st.session_state:
    st.session_state.last_effective_conf = None
if "last_used_auto_retry" not in st.session_state:
    st.session_state.last_used_auto_retry = False

st.sidebar.markdown("### Sentinel Controls")
conf_slider = st.sidebar.slider("Confidence", 0.0, 1.0, DEFAULT_UI_CONFIDENCE)
alert_toggle = st.sidebar.toggle("Enable alerts", value=True)
cooldown_slider = st.sidebar.slider("Cooldown (s)", 1, 60, 8)
custom_model_path = resolve_custom_model_path(Path(__file__).resolve().parent)
if custom_model_path is None:
    st.sidebar.warning("No se encontro modelo custom. Se usa yolov5s por defecto.")
else:
    st.sidebar.success(f"Modelo custom detectado: {custom_model_path.name}")
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
            with st.spinner("Loading model (first time) and running inference..."):
                annotated_bgr, detections_df = run_inference(image_rgb, conf_slider)
                effective_conf = conf_slider
                used_auto_retry = False
                if detections_df.empty and conf_slider > AUTO_RETRY_MIN_CONFIDENCE:
                    retry_conf = max(AUTO_RETRY_MIN_CONFIDENCE, conf_slider * AUTO_RETRY_FACTOR)
                    retry_annotated, retry_detections = run_inference(image_rgb, retry_conf)
                    if not retry_detections.empty:
                        annotated_bgr = retry_annotated
                        detections_df = retry_detections
                        effective_conf = retry_conf
                        used_auto_retry = True
            st.session_state.last_result = annotated_bgr
            st.session_state.last_detections = detections_df
            st.session_state.last_file_hash = file_hash
            st.session_state.last_conf_used = conf_slider
            st.session_state.last_effective_conf = effective_conf
            st.session_state.last_used_auto_retry = used_auto_retry
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
            if st.session_state.last_used_auto_retry:
                st.warning(
                    f"No hubo detecciones en conf={conf_slider:.2f}; se aplico auto-retry con "
                    f"conf={st.session_state.last_effective_conf:.2f}."
                )
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
