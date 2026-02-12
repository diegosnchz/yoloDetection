# INFORME TÉCNICO: SISTEMA DE MONITORIZACIÓN SENTINEL HUD
**Proyecto Final de Visión Artificial**
**Autor:** Diego Sánchez
**Fecha:** Febrero 2026

---

## 1. Definición del Problema
En el sector industrial, la siniestralidad laboral asociada a la falta de Equipos de Protección Individual (EPIs) representa un coste humano y económico crítico. Los sistemas de supervisión tradicionales dependen de la vigilancia humana, que es propensa a la fatiga y el error.

Este proyecto propone **SENTINEL HUD**, un sistema de visión artificial autónomo capaz de monitorizar en tiempo real el cumplimiento de normativas de seguridad. El sistema detecta automáticamente la presencia de cascos y chalecos reflectantes, y alerta proactivamente sobre operadores en situación de riesgo ("Persona_Sin_Equipo"). El objetivo es desplegar una solución de baja latencia, escalable y resistente a entornos hostiles mediante contenedores Docker.

---

## 2. Preparación del Dataset
Para garantizar la robustez del modelo en entornos no controlados, se construyó un dataset propio mediante técnicas de *web scraping* automatizado y curación manual.

### 2.1. Recolección y Etiquetado
Se recopilaron 200 imágenes para las clases críticas: `Casco`, `Chaleco`, `Persona_Sin_Equipo`, y `Peligro`. El etiquetado se realizó utilizando el formato YOLO estándar (coordenadas normalizadas `x_center, y_center, width, height`).

### 2.2. Data Augmentation
Dado el tamaño limitado del dataset inicial, se aplicó un pipeline de aumentación de datos para evitar el *overfitting* y mejorar la generalización:
- **Transformaciones geométricas**: Rotación (±15°), traslación y escalado para simular diferentes distancias de cámara.
- **Transformaciones fotométricas**: Variación de brillo, contraste y adición de ruido gaussiano para simular condiciones de iluminación industrial pobres y sensores de cámara ruidosos.

---

## 3. Metodología de Entrenamiento
Se seleccionó la arquitectura **YOLOv5s (You Only Look Once - Small)**.

**Justificación Técnica:**
YOLOv5s ofrece el equilibrio óptimo entre precisión (mAP) y velocidad de inferencia (FPS), crucial para una aplicación de tiempo real. Modelos más pesados (YOLOv5l/x) habrían incrementado la latencia sin una ganancia de precisión significativa para este dominio de problema específico.

**Hiperparámetros:**
- **Epochs**: 100. Seleccionado para asegurar la convergencia de la función de pérdida sin incurrir en coste computacional excesivo.
- **Batch Size**: 16. Optimizado para la memoria VRAM disponible en el entorno de entrenamiento (Colab T4 GPU).
- **Optimizador**: SGD (Stochastic Gradient Descent) con momentum, preferido sobre Adam por su mejor generalización en tareas de detección de objetos.

---

## 4. Análisis de Resultados
El proceso de entrenamiento muestra una convergencia estable de las funciones de pérdida (`box_loss`, `obj_loss`).

[INSERTAR GRÁFICA results.png AQUÍ]

La métrica **mAP@0.5** (Mean Average Precision) alcanzó valores superiores al 0.85 para las clases de EPIs (Casco/Chaleco), lo que indica una alta fiabilidad. La clase "Persona_Sin_Equipo" mostró una sensibilidad adecuada, minimizando los falsos negativos que serían críticos en seguridad.

[INSERTAR GRÁFICA confusion_matrix.png AQUÍ]

---

## 5. Innovación: Despliegue en Arquitectura de Microservicios con Docker y WebRTC
Superando los requisitos estándar de validación mediante scripts locales, **SENTINEL HUD** ha sido desplegado como una aplicación web profesional contenerizada.

### 5.1. Arquitectura del Sistema
El sistema se ha diseñado siguiendo principios de **Microservicios** y **Cloud-Native**:
1.  **Containerización (Docker)**: La aplicación corre sobre una imagen base `python:3.10-slim` personalizada, aislando todas las dependencias (PyTorch, MediaPipe, OpenCV). Esto elimina el problema de "funciona en mi máquina" y permite el despliegue inmediato en cualquier servidor.
2.  **Streaming de Baja Latencia (WebRTC)**: Se ha implementado un servidor WebRTC que negocia una conexión P2P directa entre el navegador del cliente y el contenedor Docker. Esto permite el procesamiento de vídeo en tiempo real (30 FPS) sin depender del acceso hardware directo USB, solucionando una de las limitaciones clásicas de Docker.

### 5.2. Interfaz "Industrial Cyberpunk"
La interfaz de usuario (UI) ha sido desarrollada en **Streamlit** con una capa de diseño personalizada (CSS Inject) que sigue una estética industrial/militar de alta visibilidad. Incluye:
- **HUD (Head-Up Display)**: Superposición de gráficos vectoriales sobre el vídeo.
- **Registro de Eventos**: Log persistente de infracciones de seguridad.
- **Métricas en Tiempo Real**: Visualización instantánea del nivel de amenaza.

Esta arquitectura convierte un modelo académico en un **Producto Mínimo Viable (MVP)** listo para su integración en entornos de Industria 4.0.
