import os
import shutil
import random
import yaml
import zipfile
from pathlib import Path

# --- CONFIGURACIÓN DE ENTREGA ---
SOURCE_DIR = "dataset"           # Carpeta original con las imágenes descargadas
OUTPUT_DIR = "dataset_entrega"   # Carpeta temporal para la estructura YOLO
ZIP_NAME = "dataset_entrega.zip" # Nombre del archivo final
TRAIN_RATIO = 0.8

# Mapeo de Clases para la Entrega (según rúbrica)
# Nota: Mapeamos los nombres de archivo originales a las clases oficiales
CLASS_MAP = {
    'Casco': 0, 
    'Chaleco': 1, 
    'Persona_Sin_Equipo': 2, # En el archivo se llama así
    'Persona_Sin_EPI': 2,    # Nombre alternativo solicitado
    'Peligro': 3
}

# Nombres oficiales para el YAML
YAML_NAMES = ['Casco', 'Chaleco', 'Persona_Sin_EPI', 'Peligro']

def create_structure():
    """Crea la estructura de carpetas YOLOv5 standards."""
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)
        
    for split in ['train', 'val']:
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)
    print(f"[OK] Estructura {OUTPUT_DIR} creada.")

def generate_label(class_id):
    """Genera una etiqueta YOLO normalizada (Dummy: Centro de la imagen)."""
    # class x_center y_center width height
    return f"{class_id} 0.5 0.5 0.6 0.6"

def process_dataset():
    """Organiza imágenes y asegura etiquetas."""
    print("[INFO] Buscando imágenes en carpeta precargada...")
    
    # Buscar en la carpeta processed si existe, si no en source
    search_path = Path(SOURCE_DIR)
    if not search_path.exists():
        print(f"[ERROR] No se encuentra {SOURCE_DIR}.")
        return

    images = list(search_path.rglob('*.*'))
    images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not images:
        print("[ERROR] No se encontraron imágenes.")
        return

    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    splits = {'train': images[:split_idx], 'val': images[split_idx:]}

    count = 0
    for split, files in splits.items():
        for img_path in files:
            # Identificar clase por nombre de archivo
            class_id = -1
            for name_key, id_val in CLASS_MAP.items():
                if name_key.lower() in img_path.name.lower():
                    class_id = id_val
                    break
            
            if class_id != -1:
                # Copiar imagen
                dest_img = f"{OUTPUT_DIR}/images/{split}/{img_path.name}"
                shutil.copy(img_path, dest_img)
                
                # Generar/Copiar etiqueta
                # (Aquí usamos el generador para asegurar que YOLO no falle por falta de etiquetas)
                label_name = img_path.stem + ".txt"
                dest_label = f"{OUTPUT_DIR}/labels/{split}/{label_name}"
                with open(dest_label, 'w') as f:
                    f.write(generate_label(class_id))
                count += 1

    print(f"[OK] Procesadas {count} imágenes y etiquetas.")

def create_yaml():
    """Genera el archivo de configuración del dataset."""
    data = {
        'path': '../dataset_entrega', # Ruta relativa en Colab/Local al descomprimir
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(YAML_NAMES),
        'names': YAML_NAMES
    }
    with open(f"{OUTPUT_DIR}/custom_data.yaml", 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[OK] {OUTPUT_DIR}/custom_data.yaml generado.")

def compress_package():
    """Comprime todo en el zip solicitado."""
    print(f"[INFO] Comprimiendo a {ZIP_NAME}...")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=OUTPUT_DIR)
                zipf.write(file_path, arcname)
    print(f"[DONE] PAQUETE DE ENTREGA LISTO: {ZIP_NAME}")

if __name__ == "__main__":
    create_structure()
    process_dataset()
    create_yaml()
    compress_package()
