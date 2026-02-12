import os
import shutil
import cv2
import numpy as np
from icrawler.builtin import BingImageCrawler
from pathlib import Path
import random

# Configuration
# Mapping class names to search queries
CLASSES = {
    "Casco": "industrial worker wearing safety helmet",
    "Chaleco": "industrial worker wearing safety vest",
    "Persona_Sin_Equipo": "industrial worker without safety equipment",
    "Peligro": "industrial danger warning sign"
}

NUM_IMAGES = 50
DATASET_DIR = Path("dataset")

def create_structure():
    """Create YOLO directory structure."""
    if DATASET_DIR.exists():
        print(f"Warning: {DATASET_DIR} already exists. Merging/Overwriting...")
    
    for split in ['train', 'val']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Create dataset.yaml for YOLOv5
    yaml_content = f"""
path: ../dataset  # dataset root dir
train: images/train  # train images (relative to 'path') 
val: images/val  # val images (relative to 'path')

nc: {len(CLASSES)}  # number of classes
names: {list(CLASSES.keys())}  # class names
"""
    with open("dataset.yaml", "w") as f:
        f.write(yaml_content)
    print("Created dataset.yaml")

def apply_augmentation(image):
    """Apply random data augmentation: Noise, Blur, or Rain."""
    effect = random.choice(['noise', 'blur', 'rain', 'none'])
    
    if effect == 'noise':
        # Add salt-and-pepper or gaussian noise
        row, col, ch = image.shape
        mean = 0
        var = 0.5 # Increased variance for visibility
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 30
        return np.clip(noisy, 0, 255).astype(np.uint8)
        
    elif effect == 'blur':
        # Motion blur
        size = 10
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        return cv2.filter2D(image, -1, kernel_motion_blur)
        
    elif effect == 'rain':
        # Simulated rain
        image_rain = image.copy()
        height, width, _ = image_rain.shape
        # Create rain drops
        for _ in range(int(width * height * 0.001)): # Density based on size
             x = random.randint(0, width-1)
             y = random.randint(0, height-1)
             length = random.randint(5, 15)
             cv2.line(image_rain, (x, y), (x, y+length), (200, 200, 200), 1)
        # Apply slight blur to blend
        return cv2.blur(image_rain, (2,2))
        
    return image

def download_images():
    """Download images using BingImageCrawler."""
    temp_dir = Path("temp_downloads")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    for class_name, search_query in CLASSES.items():
        print(f"\n[DOWNLOAD] Searching for: {search_query} ({class_name})...")
        save_dir = temp_dir / class_name
        save_dir.mkdir()
        
        crawler = BingImageCrawler(storage={'root_dir': str(save_dir)})
        # filters can be added here if needed, e.g., type='photo'
        crawler.crawl(keyword=search_query, max_num=NUM_IMAGES, file_idx_offset=0)
    
    return temp_dir

def process_and_organize(temp_dir):
    """Augment and split images into train/val folders."""
    print("\n[PROCESS] Organizing and Augmenting images...")
    
    for class_dir in temp_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        images = list(class_dir.glob("*"))
        random.shuffle(images)
        
        # 80/20 Split
        split_idx = int(len(images) * 0.8)
        splits = {
            'train': images[:split_idx],
            'val': images[split_idx:]
        }
        
        for split, imgs in splits.items():
            save_dir = DATASET_DIR / 'images' / split
            
            for i, img_path in enumerate(imgs):
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    # Apply augmentation to 50% of the images
                    if random.random() > 0.5:
                        img = apply_augmentation(img)
                    
                    # Save
                    ext = img_path.suffix
                    if not ext:
                        ext = ".jpg"
                        
                    filename = f"{class_name}_{i}{ext}"
                    target_path = save_dir / filename
                    cv2.imwrite(str(target_path), img)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    print(f"\n[DONE] Dataset generated at: {DATASET_DIR.absolute()}")

if __name__ == "__main__":
    try:
        import icrawler
    except ImportError:
        print("Installing missing dependency: icrawler...")
        os.system("pip install icrawler")
        
    create_structure()
    temp_dir = download_images()
    process_and_organize(temp_dir)
