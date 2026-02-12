# PROYECTO SENTINEL HUD: INFORME TÉCNICO

## 1. Definición del Problema

En el paradigma de la **Industria 4.0**, la seguridad laboral ha evolucionado de un enfoque reactivo a uno preventivo y predictivo. Los accidentes industriales relacionados con la ausencia de Equipos de Protección Individual (EPIs) representan una carga significativa tanto en términos humanos como operativos. Los sistemas de supervisión tradicionales (CCTV pasivo) dependen del factor humano, que es propenso a la fatiga y el error.

El desafío técnico abordado en este proyecto, **SENTINEL HUD**, consiste en desarrollar un sistema de visión artificial autónomo capaz de:
1.  **Detectar** en tiempo real y con alta precisión la presencia o ausencia de elementos de seguridad (cascos, chalecos) y situaciones de peligro.
2.  **Interactuar** con el operador humano mediante una interfaz de Realidad Aumentada (AR) natural, eliminando la necesidad de periféricos físicos (teclado/ratón) en entornos sucios o peligrosos.
3.  **Alertar** de forma inmediata ante violaciones de protocolos de seguridad.

La solución propuesta integra **Redes Neuronales Convolucionales (YOLOv5)** para la detección de objetos y **Redes de Grafos para Estimación de Pose (MediaPipe Hands)** para la interacción hombre-máquina (HCI), desplegando una arquitectura de borde (Edge AI) eficiente y escalable.

---

## 2. Análisis de Resultados y Justificación Arquitectónica

La implementación de una arquitectura híbrida **YOLOv5 + MediaPipe** representa un avance significativo respecto a las soluciones monolíticas tradicionales por las siguientes razones:

### A. Eficiencia Computacional y Latencia
YOLOv5 (arquitectura *single-stage detector*) ofrece un balance óptimo entre velocidad (FPS) y precisión (mAP), permitiendo su despliegue en hardware de consumo sin necesidad de servidores dedicados. Al desacoplar la lógica de detección de gestos (MediaPipe) de la detección de objetos, se optimizan los recursos: MediaPipe opera sobre *landmarks* vectoriales ligeros, liberando a la GPU para procesar la carga pesada de inferencia de YOLO.

### B. Interacción Natural (NUI - Natural User Interface)
La interfaz tipo "Minority Report" desarrollada demuestra que el control gestual es el futuro de las interfaces en planta. Al permitir que el operador "toque" virtualmente las bounding boxes para obtener metadatos (nivel de confianza, clase específica), se contextualiza la información sin saturar el campo visual, cumpliendo con los principios de diseño de **Interfaces Holísticas**.

### C. Robustez ante Ruido (Data Augmentation)
El módulo de generación de dataset sintético, que simula condiciones adversas (lluvia, desenfoque de movimiento, ruido gaussiano), garantiza que el modelo no solo memorice características limpias, sino que generalice frente a la baja calidad típica de las cámaras de seguridad industriales (CCTV), aumentando la fiabilidad del sistema en entornos reales.

**Conclusión:** SENTINEL HUD valida la viabilidad de sistemas de seguridad inteligentes de bajo coste y alta eficiencia, sentando las bases para la próxima generación de *Smart Safety Systems*.
