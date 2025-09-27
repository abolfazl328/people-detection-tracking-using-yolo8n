import cv2
import os
import shutil
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# The source YOLO-formatted dataset you want to flip
INPUT_DIR = './datasets/yolo_people_detection_dataset'
OUTPUT_DIR = './datasets/yolo_flipped_dataset'

# --- 2. PREPARE DIRECTORIES ---
print("Creating directories for the flipped dataset...")
images_in_dir = os.path.join(INPUT_DIR, 'images')
labels_in_dir = os.path.join(INPUT_DIR, 'labels')

images_out_dir = os.path.join(OUTPUT_DIR, 'images')
labels_out_dir = os.path.join(OUTPUT_DIR, 'labels')

os.makedirs(images_out_dir, exist_ok=True)
os.makedirs(labels_out_dir, exist_ok=True)


# --- 3. PROCESSING ---
image_files = sorted(os.listdir(images_in_dir)) # Sort to maintain sequence

for filename in tqdm(image_files, desc="Flipping dataset"):
    # --- Process Image ---
    img_path = os.path.join(images_in_dir, filename)
    image = cv2.imread(img_path)
    if image is None: continue
    
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)
    
    # Save the new flipped image
    new_img_filename = f"flipped_{filename}"
    cv2.imwrite(os.path.join(images_out_dir, new_img_filename), flipped_image)
    
    # --- Process Label ---
    lbl_path = os.path.join(labels_in_dir, os.path.splitext(filename)[0] + '.txt')
    new_lbl_filename = f"flipped_{os.path.splitext(filename)[0]}.txt"
    
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f_in, open(os.path.join(labels_out_dir, new_lbl_filename), 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Horizontally flipping the image only affects the x-coordinate
                # The new x_center is 1.0 minus the old x_center
                flipped_x_center = 1.0 - x_center
                
                f_out.write(f"{class_id} {flipped_x_center} {y_center} {width} {height}\n")

print("\nFlipped dataset creation complete!")