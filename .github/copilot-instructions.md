# Copilot Instructions for SENTINEL HUD

## Project Overview
SENTINEL HUD is a real-time industrial safety monitoring system using Deep Learning. It detects personal protective equipment (PPE) and risk events via a custom-trained YOLOv5 model. The system features a Streamlit dashboard for live telemetry, event history, and alert management.

## Key Components
- `core/`: Main detection logic and utilities (see `detector.py`, `utils.py`).
- `app_dashboard.py`: Streamlit dashboard for visualization and event management.
- `finalize_dataset.py`, `prepare_dataset.py`, `dataset_builder.py`: Scripts for dataset preparation and packaging.
- `dataset_entrega/`, `dataset_final/`: Custom datasets for training and validation.
- `yolov5/`: YOLOv5 code (do NOT commit changes here unless updating the model version intentionally).
- `Dockerfile`, `docker-compose.yml`: Production environment setup.

## Developer Workflows
- **Build & Run (Docker):**
  - Use `docker-compose up --build` to build and launch the full environment.
  - Access the dashboard at [http://localhost:8501](http://localhost:8501).
- **Dataset Management:**
  - Use `finalize_dataset.py` and `prepare_dataset.py` to generate and process datasets.
- **Model Training:**
  - Training is typically performed in notebooks (`Entrenamiento_YOLO.ipynb`, `SENTINEL_Entrega.ipynb`).
  - Use only the custom dataset folders for training, not the original YOLOv5 sample data.
- **Inference:**
  - Real-time inference is handled via the dashboard (`app_dashboard.py`) and `core/detector.py`.

## Project Conventions
- **Do NOT commit changes to `yolov5/` unless updating the YOLO version.**
- All custom logic should reside in `core/` or top-level scripts, not in YOLOv5 internals.
- Use SQLite for event persistence (see dashboard logic).
- Alert levels: `ALTA`, `MEDIA`, `BAJA` (see risk class rules in dashboard/core).
- Use Streamlit for all UI/UX; avoid mixing with other web frameworks.

## Integration & Patterns
- Data flows: Camera input → YOLOv5 detection → Alert/event logic (`core/`) → Dashboard display & history.
- All configuration is centralized in `config.py` and YAML files (e.g., `custom.yaml`).
- For new detection classes or alert rules, update both the dataset and the event logic in `core/` and dashboard.

## Example: Adding a New EPI Class
1. Update dataset labels and YAML config.
2. Retrain YOLOv5 model with new data.
3. Update detection/event logic in `core/detector.py` and dashboard display.

---
For questions, see `README.md` or contact the project maintainer.
