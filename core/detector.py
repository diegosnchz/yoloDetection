"""YOLO detector abstraction for Sentinel HUD."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import DEFAULT_MODEL_NAME, LOCAL_YOLO_REPO, MODEL_PATH


@dataclass
class DetectionResult:
    """Container for model output and rendered frame."""

    annotated_bgr: np.ndarray
    detections: pd.DataFrame


class YOLODetector:
    """Encapsulates YOLO model loading and inference operations."""

    def __init__(self, confidence: float = 0.4) -> None:
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.conf = confidence
        self.model.eval()

    def _load_model(self):
        """Load custom `best.pt` model when available, otherwise fallback to yolov5s."""
        if MODEL_PATH.exists():
            if LOCAL_YOLO_REPO.exists():
                return torch.hub.load(
                    str(LOCAL_YOLO_REPO.resolve()),
                    "custom",
                    path=str(MODEL_PATH),
                    source="local",
                )
            return torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(MODEL_PATH),
                trust_repo=True,
            )

        st.warning(
            "No se encontró `best.pt`; se cargará `yolov5s` por defecto. "
            "Copia el modelo custom en la raíz para detección específica de EPI."
        )
        if LOCAL_YOLO_REPO.exists():
            return torch.hub.load(
                str(LOCAL_YOLO_REPO.resolve()),
                DEFAULT_MODEL_NAME,
                source="local",
            )
        return torch.hub.load("ultralytics/yolov5", DEFAULT_MODEL_NAME, trust_repo=True)

    def detect(self, image_rgb: np.ndarray, conf_threshold: float) -> DetectionResult:
        """Run inference over an RGB frame and return annotated output and detections."""
        self.model.conf = conf_threshold
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self.model(image_bgr)
        detections = results.pandas().xyxy[0]
        filtered_detections = detections[detections["confidence"] >= conf_threshold].copy()
        return DetectionResult(annotated_bgr=image_bgr.copy(), detections=filtered_detections)


@st.cache_resource
def get_detector() -> YOLODetector:
    """Create and cache YOLO detector instance for the Streamlit app."""
    return YOLODetector()
