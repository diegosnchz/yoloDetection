# SENTINEL HUD - Proyecto Academico

**Asignatura**: Vision Artificial Avanzada
**Alumno**: Diego Sanchez

SENTINEL HUD es un sistema de monitorizacion de seguridad industrial en tiempo real basado en Deep Learning.

## Novedades implementadas (MVP Pro)

- Alertas por severidad con reglas para clases de riesgo (`ALTA`, `MEDIA`, `BAJA`).
- Cooldown configurable para evitar spam de alertas en tiempo real.
- Persistencia de eventos en SQLite (historial de timestamp, etiqueta, confianza y fuente).
- Panel historico en Streamlit con:
    - Telemetria de amenaza en vivo.
    - Conteo de alertas activas en ventana reciente.
    - Log tabular de eventos.
    - Exportacion CSV del historial.
    - Purga manual del historial desde sidebar.

## Descripcion
Este proyecto implementa un detector de **Equipos de Proteccion Individual (EPIs)** utilizando **YOLOv5** entrenado sobre un dataset personalizado. El sistema es capaz de detectar:
- Casco
- Chaleco
- Persona (Sin EPI) -> Genera Alerta
- Peligro

## Instrucciones de Ejecucion (Environment Docker)

Para garantizar la reproducibilidad del entorno y evitar conflictos de dependencias, el proyecto ha sido contenerizado.

### Prerrequisitos
- Docker Desktop instalado y corriendo.

### Pasos para desplegar:

1.  **Construir y levantar el contenedor**:
    ```bash
    docker-compose up --build
    ```

2.  **Acceder al Dashboard**:
    Abre tu navegador y visita:
    `http://localhost:8501`

3.  **Uso**:
    - Selecciona "WebRTC Stream" en el menu lateral.
    - Permite el acceso a la camara.
    - El sistema comenzara la inferencia en tiempo real.
    - Si se detectan clases de riesgo, se registraran eventos en el historial.

## Estructura del Proyecto

- `SENTINEL_Entrega.ipynb`: **[Notebook Maestro]** Informe completo y codigo de entrenamiento.
- `finalize_dataset.py`: Script de generacion del paquete de datos.
- `app_dashboard.py`: Codigo fuente de la interfaz Streamlit.
- `dataset_entrega.zip`: Dataset empaquetado para entrenamiento.
- `Dockerfile`: Configuracion del entorno de produccion.

---
*Proyecto desarrollado para la evaluacion de Febrero 2026.*
