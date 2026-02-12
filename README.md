# SENTINEL HUD: INTERFAZ DE SEGURIDAD INDUSTRIAL

SENTINEL HUD is an advanced industrial safety system that combines real-time object detection with futuristic gesture-based interaction. Designed for Industry 4.0, it monitors the use of Personal Protective Equipment (PPE) and provides a "Minority Report" style interface for operators.

## Features

- **Automated Dataset Generation**: Script to download and augment images simulating CCTV conditions (noise, blur, rain).
- **YOLOv5 PPE Detection**: Real-time detection of Helmets, Vests, Persons without PPE, and Danger signs.
- **MediaPipe Gesture Control**: Interact with detection boxes on-screen using hand gestures.
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

### 3. Run with Docker
If you prefer to use Docker to manage dependencies:
```bash
# Build the image
docker compose build

# Run the container
docker compose up
```
*Note: To see the GUI window, you need an X-Server installed on your host (like VcXsrv) and configured to allow connections.*

## Interaction
Point your webcam at safety gear (or images of them). Use your **Index Finger** to "touch" any detected object's bounding box to trigger the augmented HUD analysis.
