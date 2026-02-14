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

## Cobertura de rubrica

- [x] Definicion del problema.
- [x] Creacion/obtencion del dataset.
- [x] Proceso de entrenamiento documentado paso a paso.
- [x] Informe de resultados: datos objetivos + interpretacion.
- [x] Inferencia documentada.
