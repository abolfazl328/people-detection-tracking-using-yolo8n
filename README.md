# Human Detection and Tracking with YOLOv8

A comprehensive computer vision project for human detection and tracking using YOLOv8 with custom data augmentation and k-fold cross validation.

## 📋 Project Overview

This project implements a complete pipeline for:
- **Human detection** using YOLOv8n model
- **Object tracking** with Kalman filter
- **People counting** in defined zones
- **Robust training** with k-fold cross validation
- **Data augmentation** to overcome limited dataset size

## 🎯 Key Features

- **Complete Workflow**: From dataset preparation to real-time inference
- **Data Augmentation**: Horizontal flipping and rotation to expand dataset
- **Robust Training**: 5-fold cross validation for reliable model evaluation
- **Advanced Tracking**: Kalman filter with greedy IoU matching
- **People Counting**: Zone-based counting with consecutive frame validation
- **Real-time Performance**: 31+ FPS processing speed

## 🏗️ Project Architecture

```
project/
├── datasets/
│   ├── yolo_people_detection_dataset/     # Original converted dataset
│   ├── yolo_flipped_dataset/              # Horizontally flipped dataset  
│   ├── rotated_flipped_people_detection_dataset/  # Manually annotated rotated dataset
│   └── yolo_human_detection_final_dataset/ # Combined final dataset
├── kfold_runs/                            # Training outputs
├── greedy_outputs/                        # Inference results
├── process_people_tracking.py            # Dataset conversion
├── flip_dataset.py                       # Data augmentation
├── combine_datasets.py                   # Dataset combination
├── kfold_training.py                     # K-fold training
├── main_virtual_inference_greedy.py      # Inference & tracking
└── terminal.ipynb                        # Automated workflow
```

## 📊 Dataset Information

### Original Dataset
- **Source**: [People Tracking Dataset on Kaggle](https://www.kaggle.com/datasets/trainingdatapro/people-tracking)
- **Format**: CVAT XML annotations with PNG images
- **Initial Size**: 41 annotated frames
- **Classes**: Single class - Person (class ID: 0)
- **Annotation Type**: Bounding boxes for person detection

### Data Augmentation Strategy
1. **Format Conversion**: XML → YOLO format conversion
2. **Horizontal Flipping**: Created mirrored versions, doubling dataset size (41 → 82 images)
3. **Rotation**: Manually annotated rotated images for additional variation
4. **Dataset Combination**: Merged original + flipped + rotated datasets for enhanced training

## 🚀 Quick Start

### Prerequisites
```bash
pip install ultralytics opencv-python filterpy scikit-learn tqdm
```

### Step 1: Download Dataset
1. Visit [Kaggle Dataset](https://www.kaggle.com/datasets/trainingdatapro/people-tracking)
2. Download the dataset and extract it
3. Update the `DATASET_PATH` in `process_people_tracking.py` to point to your dataset location

### Step 2: Automated Execution
Run the complete pipeline in sequence:

```bash
# 1. Convert dataset to YOLO format
python process_people_tracking.py

# 2. Create flipped dataset for augmentation
python flip_dataset.py

# 3. Combine all datasets
python combine_datasets.py

# 4. Train model with k-fold cross validation
python kfold_training.py

# 5. Run inference with tracking
python main_virtual_inference_greedy.py
```

### Step 3: Customize for Your Use Case
- Modify `INPUT_VIDEO_PATH` in inference script for your video
- Adjust counting zone coordinates in `ZONE` variable
- Tune detection and tracking parameters as needed

## 📁 File Descriptions

### 1. Data Preparation Scripts

#### `process_people_tracking.py`
- **Purpose**: Converts Kaggle dataset from XML annotations (CVAT format) to YOLO format
- **Input**: XML annotations and image frames from Kaggle dataset
- **Output**: YOLO-formatted dataset with images and labels
- **Key Features**:
  - Parses person tracking data from XML
  - Converts bounding box coordinates to YOLO normalized format
  - Handles frames with 'outside' annotations
  - Organizes data into standard YOLO directory structure

#### `flip_dataset.py`
- **Purpose**: Data augmentation through horizontal flipping to expand small dataset
- **Input**: YOLO-formatted dataset from previous step
- **Output**: Augmented dataset with flipped images and adjusted annotations
- **Key Features**:
  - Creates mirrored versions of all images
  - Automatically adjusts bounding box x-coordinates
  - Maintains YOLO format consistency
  - Doubles dataset size for better generalization

#### `combine_datasets.py`
- **Purpose**: Merges multiple datasets into one unified dataset
- **Input**: Original, flipped, and rotated datasets
- **Output**: Combined final dataset for training
- **Key Features**:
  - Combines data from multiple sources
  - Maintains file organization
  - Creates comprehensive training set

### 2. Training Scripts

#### `kfold_training.py`
- **Purpose**: K-fold cross validation training for robust model evaluation
- **Input**: Combined final dataset
- **Output**: 5 trained models (one per fold) with comprehensive metrics
- **Key Features**:
  - 5-fold cross validation strategy
  - Automatic dataset splitting for each fold
  - YAML configuration generation
  - Best model selection based on validation performance
  - Absolute path handling to prevent pathing errors

### 3. Inference & Tracking Scripts

#### `main_virtual_inference_greedy.py`
- **Purpose**: Real-time human detection and tracking on video input
- **Input**: Trained YOLO model and input video
- **Output**: Processed video with detection, tracking, and counting
- **Key Features**:
  - YOLOv8-based human detection
  - Kalman filter for robust object tracking
  - Greedy IoU-based detection-track association
  - People counting with polygonal zone detection
  - Real-time visualization with bounding boxes and IDs

### 4. Automation

#### `terminal.ipynb`
- **Purpose**: Complete workflow automation and documentation
- **Features**:
  - Sequential execution of all processing steps
  - Progress tracking and output logging
  - Reproducible pipeline execution
  - Training and inference results documentation

## ⚙️ Configuration

### Training Parameters (kfold_training.py)
```python
K_FOLDS = 5           # Cross-validation folds
EPOCHS = 50           # Training epochs per fold
MODEL_ARCHITECTURE = 'yolov8n.pt'  # YOLO model version
IMGSZ = 640           # Image size for training
BATCH = 16            # Batch size
```

### Inference Parameters (main_virtual_inference_greedy.py)
```python
CONF_THRESHOLD = 0.3   # Detection confidence threshold
IOU_THRESHOLD = 0.05   # Intersection over Union threshold for tracking
MIN_HITS = 15          # Minimum detections to confirm a track
MAX_AGE = 10           # Maximum frames to keep lost tracks
```

### Counting Zone Configuration
```python
ZONE = np.array([[100, 500], [800, 500], [450, 200]], np.int32)
```

## 🎯 Model Performance

### Training Results (Best Performing Fold)
- **mAP50**: 0.983 (Mean Average Precision at 50% IoU)
- **Precision**: 0.942
- **Recall**: 0.95
- **mAP50-95**: 0.805
- **Training Time**: ~2 minutes per epoch on NVIDIA GTX 1650

### Inference Performance
- **Processing Speed**: 31.13 FPS
- **Tracking Stability**: High with Kalman filter smoothing
- **Counting Accuracy**: Robust with consecutive frame validation
- **Memory Usage**: Optimized for real-time performance

## 🔧 Customization Guide

### Modifying Detection Sensitivity
```python
# Increase for fewer false positives (more conservative)
CONF_THRESHOLD = 0.4

# Decrease for more detections (more sensitive)
CONF_THRESHOLD = 0.2
```

### Adjusting Tracking Behavior
```python
# Lower for faster track initiation
MIN_HITS = 10

# Higher for longer track persistence
MAX_AGE = 15
```

### Custom Counting Zone
```python
# Define custom polygon vertices [x, y]
ZONE = np.array([
    [x1, y1],
    [x2, y2], 
    [x3, y3],
    # ... more points
], np.int32)
```

### Training Configuration
```python
# Reduce for memory-constrained environments
BATCH = 8

# Adjust based on dataset size and complexity
EPOCHS = 100
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### CUDA Memory Errors
**Problem**: GPU out of memory during training
**Solution**: Reduce batch size
```python
# In kfold_training.py
batch=8  # Instead of 16
```

#### Path Configuration Issues
**Problem**: File not found errors
**Solution**: Update absolute paths in configuration
```python
# Verify these paths in each script
INPUT_DATA_DIR = '/absolute/path/to/your/dataset'
MODEL_PATH = '/absolute/path/to/trained/model.pt'
```

#### Poor Detection Performance
**Solution**:
1. Increase training epochs
2. Add more data augmentation
3. Adjust confidence thresholds
4. Verify annotation quality

#### Tracking Instability
**Solution**:
1. Adjust MIN_HITS and MAX_AGE parameters
2. Modify IoU threshold for matching
3. Check Kalman filter parameters

## 📈 Workflow Execution

### Step-by-Step Process
1. **Data Preparation**: Convert XML → YOLO format
2. **Data Augmentation**: Flip images and annotations
3. **Dataset Combination**: Merge all datasets
4. **Model Training**: 5-fold cross validation
5. **Model Selection**: Choose best performing fold
6. **Inference**: Apply detection and tracking
7. **Evaluation**: Analyze results and adjust parameters

### Expected Outputs
- **Processed Dataset**: YOLO-formatted training data
- **Trained Models**: 5 model versions from k-fold training
- **Result Videos**: Processed videos with visualization
- **Performance Metrics**: Training and validation statistics

## 🤝 Contributing

We welcome contributions to enhance this project:

### Areas for Improvement
- Implement Hungarian algorithm for optimal tracking
- Add more data augmentation techniques
- Support for multiple object classes
- Real-time webcam integration
- Export tracking data to CSV/JSON formats

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the complete pipeline
5. Submit a pull request

## 📄 License

This project is intended for educational and research purposes. Please ensure proper attribution when using or modifying this code.

## 🙏 Acknowledgments

- **Kaggle Dataset**: [People Tracking Dataset](https://www.kaggle.com/datasets/trainingdatapro/people-tracking) by TrainingDataPro
- **Ultralytics** for the YOLOv8 implementation
- **OpenCV** community for computer vision tools
- **FilterPy** for Kalman filter implementation

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Verify all file paths are correctly configured
3. Ensure all dependencies are installed
4. Create an issue in the project repository with detailed description

## 🔄 Future Enhancements

- [ ] Implement DeepSORT algorithm for improved tracking
- [ ] Add support for multiple counting zones
- [ ] Integrate with RTSP streams for live monitoring
- [ ] Add export functionality for analytics
- [ ] Web interface for configuration and monitoring

---

**Note**: This project demonstrates a complete computer vision pipeline from data preparation to deployment, showcasing robust engineering practices for real-world applications using the Kaggle People Tracking dataset.
