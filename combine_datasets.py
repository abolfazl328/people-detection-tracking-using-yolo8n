import os
import shutil
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# List of the dataset directories you want to combine
SOURCE_DIRS = ['./datasets/yolo_flipped_dataset', './datasets/yolo_people_detection_dataset', './datasets/rotated_flipped_people_detection_dataset']
FINAL_OUTPUT_DIR = './datasets/yolo_human_detection_final_dataset'

# --- 2. PREPARE DIRECTORIES ---
print(f"Creating final combined dataset directory at: {FINAL_OUTPUT_DIR}")
images_out_dir = os.path.join(FINAL_OUTPUT_DIR, 'images')
labels_out_dir = os.path.join(FINAL_OUTPUT_DIR, 'labels')
os.makedirs(images_out_dir, exist_ok=True)
os.makedirs(labels_out_dir, exist_ok=True)


# --- 3. COPYING FUNCTION ---
def copy_files(source_dir):
    source_images = os.path.join(source_dir, 'images')
    source_labels = os.path.join(source_dir, 'labels')

    for filename in tqdm(os.listdir(source_images), desc=f"Copying images from {source_dir}"):
        shutil.copy(os.path.join(source_images, filename), images_out_dir)
        
    for filename in tqdm(os.listdir(source_labels), desc=f"Copying labels from {source_dir}"):
        shutil.copy(os.path.join(source_labels, filename), labels_out_dir)

# --- 4. RUN THE PROCESS ---
for directory in SOURCE_DIRS:
    copy_files(directory)

print("\nAll datasets have been combined successfully!")