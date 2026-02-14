"""Pure inference pipeline utilities (no Streamlit dependency)."""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

MODEL_PATH_50E = Path(__file__).resolve().parents[1] / "yolov5" / "runs_academic" / "actividad1_50e" / "weights" / "best.pt"

DEFAULT_CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Casco": (0, 255, 255),
    "Chaleco": (0, 255, 0),
    "Persona_Sin_Equipo": (0, 0, 255),
    "Peligro": (255, 0, 0),
}


def ensure_cross_platform_checkpoint_loading() -> None:
    """Allow loading checkpoints saved on Windows when running in Linux containers."""
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath


def clamp_detections_to_image(detections: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    """Clamp xyxy coordinates to valid image bounds."""
    if detections.empty:
        return detections

    clamped = detections.copy()
    clamped["xmin"] = clamped["xmin"].clip(lower=0, upper=width - 1)
    clamped["ymin"] = clamped["ymin"].clip(lower=0, upper=height - 1)
    clamped["xmax"] = clamped["xmax"].clip(lower=0, upper=width - 1)
    clamped["ymax"] = clamped["ymax"].clip(lower=0, upper=height - 1)
    clamped = clamped[(clamped["xmax"] > clamped["xmin"]) & (clamped["ymax"] > clamped["ymin"])].copy()
    return clamped


def detections_to_dataframe(det_tensor: torch.Tensor, class_names: dict[int, str] | list[str]) -> pd.DataFrame:
    """Convert YOLO tensor detections [x1,y1,x2,y2,conf,cls] to DataFrame."""
    if det_tensor is None or len(det_tensor) == 0:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"])

    det_np = det_tensor.detach().cpu().numpy()
    df = pd.DataFrame(det_np, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"])
    df["class"] = df["class"].astype(int)
    if isinstance(class_names, dict):
        df["name"] = df["class"].map(class_names).fillna("unknown")
    else:
        df["name"] = df["class"].apply(lambda c: class_names[c] if 0 <= c < len(class_names) else "unknown")
    return df


def load_model_bundle(weights_path: Path | None = None) -> tuple:
    """Load model + metadata bundle required by the pure inference pipeline."""
    local_repo = Path(__file__).resolve().parents[1] / "yolov5"
    model_path = (weights_path or MODEL_PATH_50E).resolve()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo requerido: {model_path}")

    ensure_cross_platform_checkpoint_loading()
    if str(local_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(local_repo.resolve()))

    from utils.general import check_img_size  # type: ignore

    if local_repo.exists():
        model = torch.hub.load(
            str(local_repo.resolve()),
            "custom",
            path=str(model_path),
            source="local",
            autoshape=False,
        )
    else:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(model_path), trust_repo=True, autoshape=False)

    model.to(device)
    model.eval()
    if hasattr(model, "stride"):
        stride_attr = model.stride
        stride = int(max(stride_attr)) if hasattr(stride_attr, "__iter__") else int(stride_attr)
    else:
        stride = 32
    imgsz = check_img_size(640, s=stride)
    class_names = model.names if hasattr(model, "names") else {}
    return model, device, model_path, int(imgsz), stride, class_names


def run_inference_detailed(
    image_bgr_or_rgb: np.ndarray,
    model_bundle: tuple,
    conf_slider: float,
    iou: float = 0.45,
    class_colors_bgr: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run detect.py-equivalent inference and return annotated image + filtered/raw detections."""
    model, device, _model_path, imgsz, stride, class_names = model_bundle
    image_rgb = image_bgr_or_rgb
    h, w = image_rgb.shape[:2]
    print(f"[DEBUG] inference_image_shape={image_rgb.shape}")

    from utils.augmentations import letterbox  # type: ignore
    from utils.general import non_max_suppression, scale_boxes  # type: ignore

    letterboxed = letterbox(image_rgb, new_shape=imgsz, stride=stride, auto=True)[0]
    im = letterboxed.transpose((2, 0, 1))
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float() / 255.0
    if im.ndim == 3:
        im = im.unsqueeze(0)

    with torch.no_grad():
        pred = model(im, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.001, iou_thres=iou, max_det=1000)

    det = pred[0]
    if det is not None and len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image_rgb.shape).round()

    raw_detections = detections_to_dataframe(det, class_names)
    print(f"[DEBUG] raw_detections={len(raw_detections)}")

    raw_detections = clamp_detections_to_image(raw_detections, w, h)
    filtered = raw_detections[raw_detections["confidence"] >= conf_slider].copy()

    color_map = class_colors_bgr or DEFAULT_CLASS_COLORS_BGR
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for _, det_row in filtered.iterrows():
        x1, y1, x2, y2 = int(det_row["xmin"]), int(det_row["ymin"]), int(det_row["xmax"]), int(det_row["ymax"])
        label = str(det_row["name"])
        confidence = float(det_row["confidence"])
        color = color_map.get(label, (255, 255, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.2f}"
        text_ymin = max(0, y1 - 22)
        cv2.rectangle(annotated, (x1, text_ymin), (x1 + 8 * len(text), y1), color, -1)
        cv2.putText(annotated, text, (x1 + 4, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated, filtered, raw_detections


def run_inference(
    image_bgr_or_rgb: np.ndarray,
    model,
    conf_slider: float,
    iou: float = 0.45,
) -> pd.DataFrame:
    """Pure inference API required by tests and non-UI callers."""
    _annotated, filtered, _raw = run_inference_detailed(
        image_bgr_or_rgb=image_bgr_or_rgb,
        model_bundle=model,
        conf_slider=conf_slider,
        iou=iou,
    )
    return filtered

