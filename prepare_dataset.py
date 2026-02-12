import os
import shutil
import random
import yaml
import zipfile
from pathlib import Path

# Configuration
SOURCE_DIR = "dataset"
OUTPUT_DIR = "dataset_final"
TRAIN_RATIO = 0.8
# Map class names to IDs for YOLO
CLASS_MAP = {
    'Casco': 0, 
    'Chaleco': 1, 
    'Persona_Sin_Equipo': 2, 
    'Peligro': 3
}

def create_structure():
    """Crea la estructura de carpetas YOLOv5"""
    for split in ['train', 'val']:
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)
    print(f"[OK] Estructura creada en {OUTPUT_DIR}/")

def generate_dummy_label(class_name, file_path):
    """Generates a dummy label file assuming the object is in the center."""
    # Format: class x_center y_center width height
    # Box covering 60% of image in center
    class_id = CLASS_MAP.get(class_name, 0)
    return f"{class_id} 0.5 0.5 0.6 0.6"

def organize_dataset():
    """Mueve y divide las imágenes y genera labels sinteticos"""
    image_paths = []
    # Using specific class folders if they exist (based on dataset_builder structure)
    # dataset/images/train/Casco_0.jpg
    
    # But wait, dataset_builder puts them in dataset/images/train directly
    # Filenames are like "Casco_123.jpg"
    
    source_path = Path(SOURCE_DIR) / 'images'
    
    # Gather from existing train/val in source (created by dataset_builder)
    all_images = list(source_path.rglob('*.*'))
    # Filter only images
    all_images = [p for p in all_images if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"[INFO] Encontradas {len(all_images)} imagenes fuente.")
    
    random.shuffle(all_images)
    split_idx = int(len(all_images) * TRAIN_RATIO)
    splits = {
        'train': all_images[:split_idx],
        'val': all_images[split_idx:]
    }
    
    for split, files in splits.items():
        for img_path in files:
            # 1. Copy Image
            dst_img = f"{OUTPUT_DIR}/images/{split}/{img_path.name}"
            shutil.copy(img_path, dst_img)
            
            # 2. Determine Class from Filename
            # Filename format expected: ClassName_Index.jpg
            # If not matching, default to 0
            found_class = None
            for cls_name in CLASS_MAP.keys():
                if cls_name in img_path.name:
                    found_class = cls_name
                    break
            
            if found_class:
                # 3. Generate Label
                label_content = generate_dummy_label(found_class, img_path)
                label_name = img_path.stem + ".txt"
                dst_label = f"{OUTPUT_DIR}/labels/{split}/{label_name}"
                with open(dst_label, 'w') as f:
                    f.write(label_content)
    
    print(f"[OK] {len(splits['train'])} imagenes procesadas para TRAIN")
    print(f"[OK] {len(splits['val'])} imagenes procesadas para VAL")

def create_yaml():
    data = {
        'path': '../dataset_final', 
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_MAP),
        'names': list(CLASS_MAP.keys())
    }
    
    with open(f"{OUTPUT_DIR}/custom_data.yaml", 'w') as f:
        yaml.dump(data, f)
    print("[OK] Archivo custom_data.yaml generado")

def zip_dataset():
    zip_name = "dataset_final.zip"
    print(f"[INFO] Comprimiendo a {zip_name}...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=OUTPUT_DIR)
                zipf.write(file_path, arcname)
    print(f"[DONE] LISTO: {zip_name} generado exitosamente.")

if __name__ == "__main__":
    if not Path(SOURCE_DIR).exists():
        print(f"[ERROR] No se encuentra la carpeta '{SOURCE_DIR}'")
    else:
        # Clean output dir
        if Path(OUTPUT_DIR).exists():
            shutil.rmtree(OUTPUT_DIR)
            
        create_structure()
        organize_dataset()
        create_yaml()
        zip_dataset()
