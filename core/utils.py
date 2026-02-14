"""Utility helpers for Sentinel Streamlit UI and post-processing."""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import pandas as pd

from config import BOX_COLORS_BGR, CYBERPUNK_COLORS_HEX, SEVERITY_HIGH_KEYWORDS, SEVERITY_MEDIUM_KEYWORDS


def get_cyberpunk_css() -> str:
    """Return CSS styles used to preserve the Industrial/Cyberpunk visual theme."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {{
        background: radial-gradient(1200px 500px at 50% -10%, #1a1a1a 0%, {CYBERPUNK_COLORS_HEX['bg_primary']} 35%, #000 70%);
        color: {CYBERPUNK_COLORS_HEX['text_primary']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background: {CYBERPUNK_COLORS_HEX['bg_secondary']};
        border-right: 1px solid {CYBERPUNK_COLORS_HEX['border']};
    }}

    .title-wrap {{
        padding: 8px 0 16px 0;
    }}

    .title-main {{
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
    }}

    .title-sub {{
        color: {CYBERPUNK_COLORS_HEX['text_secondary']};
        font-size: 0.95rem;
        margin-top: 4px;
    }}

    .card {{
        border: 1px solid {CYBERPUNK_COLORS_HEX['border']};
        background: linear-gradient(180deg, rgba(18,18,18,0.95), rgba(10,10,10,0.95));
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }}

    .card h3 {{
        margin: 0;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #e5e7eb;
    }}

    div[data-testid="stMetric"] {{
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        background: #0f0f10;
        padding: 10px;
    }}

    div.stButton > button {{
        background: #f5f5f5;
        color: #0b0b0b;
        border: 1px solid #f5f5f5;
        border-radius: 10px;
        font-weight: 600;
    }}

    div.stButton > button:hover {{
        background: #ffffff;
        border-color: #ffffff;
    }}

    .stDownloadButton > button {{
        border-radius: 10px;
    }}
    </style>
    """


def file_md5(content: bytes) -> str:
    """Compute file hash used for result caching in session state."""
    return hashlib.md5(content).hexdigest()


def severity_for_label(label: str) -> str:
    """Map label names to event severity level."""
    label_lower = label.lower()
    if any(keyword in label_lower for keyword in SEVERITY_HIGH_KEYWORDS):
        return "ALTA"
    if any(keyword in label_lower for keyword in SEVERITY_MEDIUM_KEYWORDS):
        return "MEDIA"
    return "BAJA"


def draw_detection_boxes(image_bgr, detections: pd.DataFrame) -> Any:
    """Draw bounding boxes and labels on top of a BGR image."""
    annotated = image_bgr.copy()
    for _, det in detections.iterrows():
        x1, y1, x2, y2 = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
        label = str(det["name"])
        confidence = float(det["confidence"])
        severity = severity_for_label(label)
        color = BOX_COLORS_BGR.get(severity, BOX_COLORS_BGR["default"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1 - 22), (x1 + 8 * len(text), y1), color, -1)
        cv2.putText(annotated, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return annotated


def format_detection_table(detections: pd.DataFrame) -> pd.DataFrame:
    """Build user-friendly detection table for Streamlit rendering."""
    view_df = detections[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]].copy()
    view_df["confidence"] = (view_df["confidence"] * 100).round(2).astype(str) + "%"
    return view_df
