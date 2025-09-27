import os
import shutil
from sklearn.model_selection import KFold
from ultralytics import YOLO
import yaml

# --- 1. CONFIGURATION ---
INPUT_DATA_DIR = './datasets/yolo_human_detection_final_dataset'
K_FOLDS = 5
EPOCHS = 50
MODEL_ARCHITECTURE = 'yolov8n.pt'

# --- 2. PREPARE DATA ---
images_dir = os.path.join(INPUT_DATA_DIR, 'images')
labels_dir = os.path.join(INPUT_DATA_DIR, 'labels')
all_files = sorted(os.listdir(images_dir))
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# --- 3. K-FOLD TRAINING LOOP ---
for fold, (train_indices, val_indices) in enumerate(kf.split(all_files)):
    print(f"===== FOLD {fold + 1}/{K_FOLDS} =====")
    
    fold_dir = f'./kfold_v1_fold_{fold + 1}'
    for split in ['train', 'val']:
        os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)

    for idx in train_indices:
        filename = all_files[idx]
        shutil.copy(os.path.join(images_dir, filename), os.path.join(fold_dir, 'images', 'train'))
        shutil.copy(os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt'), os.path.join(fold_dir, 'labels', 'train'))

    for idx in val_indices:
        filename = all_files[idx]
        shutil.copy(os.path.join(images_dir, filename), os.path.join(fold_dir, 'images', 'val'))
        shutil.copy(os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt'), os.path.join(fold_dir, 'labels', 'val'))

    # --- FIX: USE ABSOLUTE PATHS IN THE YAML FILE ---
    # This resolves the pathing error by providing the full, unambiguous location of the data.
    train_path_abs = os.path.abspath(os.path.join(fold_dir, 'images', 'train'))
    val_path_abs = os.path.abspath(os.path.join(fold_dir, 'images', 'val'))

    yaml_data = {
        'train': train_path_abs,
        'val': val_path_abs,
        'nc': 1,
        'names': ['person']
    }
    yaml_path = os.path.join(fold_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    # --- END OF FIX ---

    model = YOLO(MODEL_ARCHITECTURE)
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        project='kfold_runs',
        name=f'fold_v1_{fold + 1}'
    )
    
print("\nK-Fold Cross-Validation complete!")