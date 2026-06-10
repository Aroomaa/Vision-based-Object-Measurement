# Measurement System 
(Mask R-CNN + Camera Calibration)

## Overview

This project implements a full computer vision pipeline for **detecting, segmenting, and measuring real-world dimensions of desired object, in this case envelope is selected** using:

- Camera calibration (lens distortion removal)
- Mask R-CNN instance segmentation
- Pixel-to-mm conversion using a reference object
- Automated width/height measurement system

The system outputs **real-world measurements (mm)** from images using deep learning + classical vision.

---

## Key Features

- Camera calibration using checkerboard images
- Image undistortion for geometric accuracy
- Envelope detection using Mask R-CNN (PyTorch)
- Pixel-to-mm conversion using reference ID card
- Automatic width & height measurement
- Visualization of results with annotations
- Full dataset pipeline (train/val/test split)
- Evaluation metrics and error analysis

---

## Project Structure

xis_assessment/
├── calibration/
├── dataset/
├── models/
├── measurement/
├── docs/
├── requirements.txt
└── README.md

more detailed in `SETUP.md`


---

## Object Information

- **Object:** Envelope  
- **Dimensions:** 139 mm × 90 mm  
- **Task:** Instance Segmentation + Measurement  
- **Framework:** PyTorch (Mask R-CNN)

---

## Pipeline Overview

### 1. Camera Calibration
- Checkerboard-based calibration using OpenCV
- Computes:
  - Camera matrix
  - Distortion coefficients
- Removes lens distortion for accurate measurement

### 2. Dataset Preparation
- Images captured using smartphone camera
- Conditions:
  - Different lighting
  - Multiple angles
  - Varied backgrounds
- Annotation tool: CVAT (COCO format)
- Split:
  - Train: 70%
  - Validation: 20%
  - Test: 10%

### 3. Model Training
- Model: Mask R-CNN (ResNet-50-FPN)
- Framework: PyTorch
- Task: Instance segmentation of envelope

**Hyperparameters:**

| Parameter | Value |
|----------|------|
| Epochs | 50 |
| Batch Size | 2 |
| Learning Rate | 0.005 |

---

## Measurement System

### Reference Object
- ID Card (85.6 mm × 54.0 mm)
- Used for pixel-to-mm conversion
- Detected using HSV color segmentation

### Method
1. Detect reference card
2. Compute pixels per mm
3. Detect envelope using Mask R-CNN
4. Extract mask and contour
5. Fit rotated bounding box (`minAreaRect`)
6. Convert pixels → mm

---

## Results

### Model Performance

| Metric | Value |
|--------|------|
| mAP@0.5 | 1.00 |
| mAP@0.5:0.95 | 0.9723 |
| Mean IoU | 0.957 |
| Precision | 1.00 |
| Recall | 1.00 |

---

### Measurement Accuracy

| Metric | Width | Height |
|--------|------|--------|
| MAE | 3.23 mm | 7.01 mm |
| MPE | 3.59% | 5.04% |

---


## Limitations

- Requires reference card in every image
- Sensitive to lighting and color similarity (HSV detection)
- Assumes flat object plane
- Accuracy decreases with increased camera distance

---

## Installation

### Clone Repository
```bash
git clone 
cd xis_assessment
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Camera Calibration

```bash
python calibration/calibrate.py
```
### 2. Undistort Dataset
```bash
python calibration/undistort_dataset.py
```

### 3. Train Model
```bash
python models/train.py
```
### 4. Evaluate Model
```bash
python models/evaluate.py
```

### 5. Run Measurement Pipeline
```bash
python measurement/measurement.py
```

## Output
Trained model: `models/maskrcnn_final.pth`
Loss curves: `models/loss_curves.png`
Evaluation metrics: `models/evaluation_results.json`
Measurement results: `measurement/outputs/`

### Technologies Used
- Python
- PyTorch
- Torchvision (Mask R-CNN)
- OpenCV
- NumPy
- Matplotlib
- CVAT (annotation tool)