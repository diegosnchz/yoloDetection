"""Main Streamlit UI for Sentinel HUD."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import DEFAULT_CONFIDENCE_THRESHOLD, PAGE_SUBTITLE, PAGE_TITLE
from core.detector import get_detector
from core.utils import (
    draw_detection_boxes,
    file_md5,
    format_detection_table,
    get_cyberpunk_css,
    severity_for_label,
)


st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_cyberpunk_css(), unsafe_allow_html=True)


if "event_log" not in st.session_state:
    st.session_state.event_log: list[dict[str, str | float]] = []
if "last_alert_by_key" not in st.session_state:
    st.session_state.last_alert_by_key: dict[str, float] = {}
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_detections" not in st.session_state:
    st.session_state.last_detections = pd.DataFrame()
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None
if "last_conf_used" not in st.session_state:
    st.session_state.last_conf_used = None


def insert_event(label: str, confidence: float, severity: str, source: str = "upload") -> None:
    """Insert an event into in-memory session log."""
    timestamp = time.time()
    st.session_state.event_log.append(
        {
            "ts_unix": timestamp,
            "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "severity": severity,
            "label": label,
            "confidence": confidence,
            "source": source,
        }
    )


def get_events_df(limit: int = 20) -> pd.DataFrame:
    """Return the latest events as DataFrame."""
    if not st.session_state.event_log:
        return pd.DataFrame(columns=["ts_iso", "severity", "label", "confidence", "source"])
    events = pd.DataFrame(st.session_state.event_log)
    return events.sort_values("ts_unix", ascending=False).head(limit)[
        ["ts_iso", "severity", "label", "confidence", "source"]
    ]


def get_event_metrics() -> dict[str, str | int | pd.DataFrame]:
    """Compute telemetry metrics from current session events."""
    if not st.session_state.event_log:
        return {
            "total_events": 0,
            "high_last_60s": 0,
            "threat": "LOW",
            "severity_counts": pd.DataFrame(columns=["severity", "count"]),
        }

    events = pd.DataFrame(st.session_state.event_log)
    now = time.time()
    high_last_60s = len(events[(events["severity"] == "ALTA") & (events["ts_unix"] >= now - 60)])
    high_last_5m = len(events[(events["severity"] == "ALTA") & (events["ts_unix"] >= now - 300)])
    medium_last_5m = len(events[(events["severity"] == "MEDIA") & (events["ts_unix"] >= now - 300)])

    if high_last_5m > 0:
        threat = "HIGH"
    elif medium_last_5m > 2:
        threat = "MEDIUM"
    else:
        threat = "LOW"

    severity_counts = (
        events.groupby("severity", as_index=False).size().rename(columns={"size": "count"})
    )

    return {
        "total_events": len(events),
        "high_last_60s": high_last_60s,
        "threat": threat,
        "severity_counts": severity_counts,
    }


st.sidebar.markdown("### Sentinel Controls")
conf_slider = st.sidebar.slider("Confidence", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD)
alert_toggle = st.sidebar.toggle("Enable alerts", value=True)
cooldown_slider = st.sidebar.slider("Cooldown (s)", 1, 60, 8)
if st.sidebar.button("Purge event history"):
    st.session_state.event_log = []
    st.sidebar.success("History cleared")

st.markdown(
    f"""
    <div class="title-wrap">
        <p class="title-main">{PAGE_TITLE}</p>
        <p class="title-sub">{PAGE_SUBTITLE}</p>
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
        current_hash = file_md5(image_bytes)
        same_input = (
            st.session_state.last_file_hash == current_hash
            and st.session_state.last_conf_used == conf_slider
            and st.session_state.last_result is not None
        )

        if not same_input:
            image_rgb: np.ndarray = np.array(Image.open(uploaded_file).convert("RGB"))
            detector = get_detector()
            with st.spinner("Loading model (first time) and running inference..."):
                detection_result = detector.detect(image_rgb, conf_slider)
                annotated_bgr = draw_detection_boxes(
                    detection_result.annotated_bgr,
                    detection_result.detections,
                )

            st.session_state.last_result = annotated_bgr
            st.session_state.last_detections = detection_result.detections
            st.session_state.last_file_hash = current_hash
            st.session_state.last_conf_used = conf_slider
        else:
            st.info("Using cached result for this image and confidence.")

        detections_df = st.session_state.last_detections
        if alert_toggle and not detections_df.empty:
            for _, det in detections_df.iterrows():
                label = str(det["name"])
                confidence = float(det["confidence"])
                severity = severity_for_label(label)
                if severity in {"ALTA", "MEDIA"}:
                    key = f"{severity}:{label}:{current_hash}"
                    now_t = time.time()
                    last_t = st.session_state.last_alert_by_key.get(key, 0.0)
                    if now_t - last_t >= cooldown_slider:
                        insert_event(label=label, confidence=confidence, severity=severity)
                        st.session_state.last_alert_by_key[key] = now_t

    if st.session_state.last_result is not None:
        st.image(st.session_state.last_result, channels="BGR", use_container_width=True)
        if st.session_state.last_detections.empty:
            st.info("No detections above confidence threshold.")
        else:
            st.dataframe(
                format_detection_table(st.session_state.last_detections),
                use_container_width=True,
                hide_index=True,
            )

with col_right:
    metrics = get_event_metrics()
    events_df = get_events_df(limit=20)

    st.markdown('<div class="card"><h3>Telemetry</h3></div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Threat level", str(metrics["threat"]))
    m2.metric("Active alerts (60s)", str(metrics["high_last_60s"]))
    st.metric("Total events", str(metrics["total_events"]))

    st.markdown('<div class="card"><h3>Severity mix</h3></div>', unsafe_allow_html=True)
    severity_counts = metrics["severity_counts"]
    if isinstance(severity_counts, pd.DataFrame) and not severity_counts.empty:
        st.bar_chart(severity_counts.set_index("severity"))
    else:
        st.info("No events yet.")

    st.markdown('<div class="card"><h3>Event log</h3></div>', unsafe_allow_html=True)
    if events_df.empty:
        st.caption("Waiting for events...")
    else:
        display_df = events_df.copy()
        display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Export events CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="sentinel_events.csv",
            mime="text/csv",
            use_container_width=True,
        )
