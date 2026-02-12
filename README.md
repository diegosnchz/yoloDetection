# SENTINEL HUD: INTERFAZ DE SEGURIDAD INDUSTRIAL

SENTINEL HUD is an advanced industrial safety system that combines real-time object detection with futuristic gesture-based interaction. Designed for Industry 4.0, it monitors the use of Personal Protective Equipment (PPE) and provides a "Minority Report" style interface for operators.

## Key Features
- **Professional Command Dashboard**: Web-based interface built with Streamlit for real-time monitoring.
- **YOLOv5 Integration**: High-precision detection of PPE (Helmets, Vests) and danger zones.
- **Gesture Control**: Interactive HUD elements controlled by hand gestures via MediaPipe.
- **Security Analytics**: Real-time metrics and historical log of security breaches.
- **Dockerized Environment**: Ready for industrial deployment with 1-click setup.
- **Industrial HUD**: Futuristic head-up display with neon aesthetics and holographic analysis details.
- **Security Alerts**: Visual red-flash alerts when safety breaches are detected.

## Project Structure

- `dataset_builder.py`: Python script for dataset creation and augmentation.
- `Entrenamiento_YOLO.ipynb`: Jupyter Notebook for training the model on Google Colab.
- `app_hud.py`: Core application for real-time inference and HUD interaction.
- `dataset.yaml`: Configuration for YOLO training.
- `Reporte_Final.md`: Technical documentation of the project.

## Installation

1. Install local dependencies:
   ```bash
   pip install icrawler opencv-python mediapipe torch torchvision numpy
   ```

2. Clone YOLOv5 (required for training and local hub loading):
   ```bash
   git clone https://github.com/ultralytics/yolov5
   ```

## Usage

### 1. Build Dataset
Run the builder to scrape images and prepare the YOLO structure:
```bash
python dataset_builder.py
```

### 2. Train Model
Upload `Entrenamiento_YOLO.ipynb` and the zipped `dataset` folder to Google Colab. Run all cells to obtain `best.pt`.

### 3. Run Professional Dashboard
To launch the full industrial interface:
```bash
python -m streamlit run app_dashboard.py
```

### 4. Run with Docker
If you prefer to use the containerized version:
```bash
docker compose build
docker compose up
```

## Interaction
1. **Visual Monitoring**: See real-time detections and security metrics on the dashboard.
2. **Gesture HUD**: Point your **Index Finger** at detection boxes in the video feed to trigger detailed HUD info.
3. **Security Log**: Check the sidebar and status table for active alerts.
