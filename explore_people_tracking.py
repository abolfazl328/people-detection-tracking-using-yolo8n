import cv2
import xml.etree.ElementTree as ET
import os

# --- 1. CONFIGURATION ---
DATASET_PATH = "../../human_detection&trackin_dataset"
IMAGE_FOLDER = "images" # Assuming you've extracted frames here
ANNOTATION_FILENAME = "annotations.xml"
FRAME_TO_VERIFY = 20 # Which frame number to check

# --- 2. CONSTRUCT PATHS ---
image_path = os.path.join(DATASET_PATH, IMAGE_FOLDER, f"frame_{FRAME_TO_VERIFY:06d}.png")
annotations_path = os.path.join(DATASET_PATH, ANNOTATION_FILENAME)

# --- 3. PARSE XML AND DRAW BOXES ---
frame = cv2.imread(image_path)
if frame is not None:
    try:
        tree = ET.parse(annotations_path)
        root = tree.getroot()

        # Find all <box> tags that belong to the specific frame
        for track in root.findall('.//track[@label="person"]'):
            for box in track.findall(f'.//box[@frame="{FRAME_TO_VERIFY}"]'):
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                # Convert to integers for drawing
                x1, y1, x2, y2 = int(xtl), int(ytl), int(xbr), int(ybr)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save the verification image
        output_image_path = './people_tracking_verification.jpg'
        cv2.imwrite(output_image_path, frame)
        print(f"Saved sample verification frame to {output_image_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print(f"Error: Could not load image at {image_path}. Make sure frames are extracted.")