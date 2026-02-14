# Checklist de Cumplimiento - Actividad 1 (YOLOv5)

Fecha de verificacion: 2026-02-14

## Entregables de formato

- [x] `YOLOv5_alumnos.ipynb` creado.
- [x] `YOLOv5_alumnos.pdf` creado.
- [x] Notebook y PDF incluyen secciones alineadas con la rubrica.

## Pautas de elaboracion

- [x] Problema definido para deteccion de objetos (seguridad industrial con EPI).
- [x] Dataset en formato YOLO con estructura `images/train`, `images/val`, `labels/train`, `labels/val`.
- [x] Uso de YOLOv5 (repositorio clonado localmente en `yolov5/`).
- [x] Entrenamiento ejecutado con `custom_data.yaml`.
- [x] Informe de validacion con metricas objetivas y analisis.
- [x] Inferencia ejecutada y documentada con salidas guardadas.

## Evidencias tecnicas (archivos)

- Configuracion dataset: `dataset_final/custom_data.yaml`
- Entrenamiento: `yolov5/runs_academic/actividad1_10e/`
- Pesos finales: `yolov5/runs_academic/actividad1_10e/weights/best.pt`
- Curvas y matriz: `yolov5/runs_academic/actividad1_10e/results.png`, `yolov5/runs_academic/val_actividad1/confusion_matrix.png`
- Inferencia: `yolov5/runs_academic/inferencia_actividad1_val/`

## Actualizacion 50 epocas (evidencia adicional)

- [x] Entrenamiento 50e ejecutado sin sobrescribir 10e.
  - Run: `yolov5/runs_academic/actividad1_50e/`
  - Pesos: `yolov5/runs_academic/actividad1_50e/weights/best.pt`
  - Parametros clave: `img=640`, `batch=16`, `epochs=50`, `seed=0`, `device=0`
  - Log de consola: `yolov5/runs_academic/train_actividad1_50e_log.txt`
- [x] Validacion fija para comparacion 10e vs 50e (batch y thresholds constantes).
  - 10e bs16: `yolov5/runs_academic/val_actividad1_10e_bs16/`
  - 10e log: `yolov5/runs_academic/val_actividad1_10e_bs16_log.txt`
  - 50e bs16: `yolov5/runs_academic/val_actividad1_50e/`
  - 50e log: `yolov5/runs_academic/val_actividad1_50e_log.txt`
- [x] Comparacion documentada en notebook y PDF (global + por clase) usando `results.csv` y logs reales.
- [x] Inferencia externa con modelo 50e documentada.
  - Salidas: `yolov5/runs_academic/inferencia_actividad1_50e_new_txt/`
  - Etiquetas predichas: `yolov5/runs_academic/inferencia_actividad1_50e_new_txt/labels/`

## Cobertura de rubrica

- [x] Definicion del problema.
- [x] Creacion/obtencion del dataset.
- [x] Proceso de entrenamiento documentado paso a paso.
- [x] Informe de resultados: datos objetivos + interpretacion.
- [x] Inferencia documentada.
