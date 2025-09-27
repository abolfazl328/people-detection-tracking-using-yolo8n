import cv2
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import shutil
from collections import defaultdict

# --- 1. CONFIGURATION ---
DATASET_PATH = "../../human_detection&trackin_dataset"
IMAGE_FOLDER = "images"
ANNOTATION_FILENAME = "annotations.xml"
OUTPUT_DIR = "./datasets/yolo_people_detection_dataset"

# --- 2. PREPARE DIRECTORIES ---
images_out_dir = os.path.join(OUTPUT_DIR, 'images')
labels_out_dir = os.path.join(OUTPUT_DIR, 'labels')
os.makedirs(images_out_dir, exist_ok=True)
os.makedirs(labels_out_dir, exist_ok=True)

# --- 3. PARSE XML ANNOTATIONS ---
print("Parsing annotations.xml...")
annotations_path = os.path.join(DATASET_PATH, ANNOTATION_FILENAME)
tree = ET.parse(annotations_path)
root = tree.getroot()

# Get image dimensions from XML meta tag
meta = root.find('meta/task/original_size')
frame_width = int(meta.find('width').text)
frame_height = int(meta.find('height').text)

# Group all bounding boxes by their frame number
annotations_by_frame = defaultdict(list)
for track in tqdm(root.findall('.//track[@label="person"]'), desc="Reading Tracks"):
    for box in track.findall('box'):
        # Ensure the box is not marked as 'outside' the frame
        if int(box.get('outside')) == 1:
            continue
            
        frame_num = int(box.get('frame'))
        xtl = float(box.get('xtl'))
        ytl = float(box.get('ytl'))
        xbr = float(box.get('xbr'))
        ybr = float(box.get('ybr'))
        annotations_by_frame[frame_num].append([xtl, ytl, xbr, ybr])

print(f"Found annotations for {len(annotations_by_frame)} frames.")

# --- 4. CREATE YOLO DATASET ---
print("Creating YOLO formatted dataset...")
for frame_num, boxes in tqdm(annotations_by_frame.items(), desc="Processing Frames"):
    
    # Define source and destination paths based on your findings
    source_img_filename = f"frame_{frame_num:06d}.png"
    dest_img_filename = f"people_tracking_{frame_num:06d}.jpg" # Convert to JPG for consistency
    label_filename = f"people_tracking_{frame_num:06d}.txt"

    source_img_path = os.path.join(DATASET_PATH, IMAGE_FOLDER, source_img_filename)
    
    # Check if the source image actually exists
    if not os.path.exists(source_img_path):
        continue

    # Copy and rename the image file (and convert to JPG)
    image = cv2.imread(source_img_path)
    if image is not None:
        cv2.imwrite(os.path.join(images_out_dir, dest_img_filename), image)

        # Create the corresponding YOLO label file
        with open(os.path.join(labels_out_dir, label_filename), 'w') as f:
            for box in boxes:
                x1, y1, x2, y2 = box
                
                # Convert coordinates to YOLO format
                box_w = x2 - x1
                box_h = y2 - y1
                x_center = (x1 + box_w / 2) / frame_width
                y_center = (y1 + box_h / 2) / frame_height
                width_norm = box_w / frame_width
                height_norm = box_h / frame_height
                
                # Class ID for 'person' is 0
                f.write(f"0 {x_center} {y_center} {width_norm} {height_norm}\n")

print(f"\nProcessing complete. New dataset is in: {OUTPUT_DIR}")