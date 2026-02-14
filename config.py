"""Global configuration constants for Sentinel HUD."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
MODEL_FILENAME: str = "best.pt"
MODEL_PATH: Path = PROJECT_ROOT / MODEL_FILENAME
LOCAL_YOLO_REPO: Path = PROJECT_ROOT / "yolov5"

DEFAULT_MODEL_NAME: str = "yolov5s"
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.45

SEVERITY_HIGH_KEYWORDS: tuple[str, ...] = ("sin_equipo", "sin_epi", "danger", "peligro")
SEVERITY_MEDIUM_KEYWORDS: tuple[str, ...] = ("persona",)

CYBERPUNK_COLORS_HEX: dict[str, str] = {
    "bg_primary": "#0a0a0a",
    "bg_secondary": "#0b0b0b",
    "text_primary": "#f5f5f5",
    "text_secondary": "#9ca3af",
    "border": "#222222",
    "danger": "#ff0033",
    "warning": "#ffc857",
    "accent": "#00f5ff",
}

BOX_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "default": (255, 255, 255),
    "ALTA": (0, 0, 255),
    "MEDIA": (0, 200, 255),
    "BAJA": (120, 120, 120),
}

PAGE_TITLE: str = "SENTINEL Vision"
PAGE_SUBTITLE: str = "Upload images, detect risk patterns, and track events in real time."
