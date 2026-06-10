# Setup and Installation Guide

### System Requirements

| **Component** | **Requirement** |
|---------------|---------------- |
| Python | 3.9 or higher |
| Operating System | Windows 10/11 |
| RAM | 8 GB minimum (16 GB recommended for training) |
| GPU | Optional  |
| Disk space | ~2 GB (images + model weights)

### Installation
```
git clone 
cd xis_assessment
```

### Install Dependencies

`pip install -r requirements.txt`


### Procedure

**Step 01**

Camera calibration
Print corner detection results
Print camera matrix and distortion coefficients
Print reprojection error
Save calibration parameters

Ensure checkerboard images are in `calibration/Images/`
```bash
python calibration/calibrate.py
```
Output: `calibration/calibration_params.npz`

**Step 02**

Verfy undistortion
Comapre distortion and undistortion results

Ensure raw envelope images are in `dataset/Raw/`
Output: 100 undistorted images in `dataset/undistorted/`

**Step 03**

Get aanotation `.json` file
Split daatset into train, validation, test
Output: Train/val/test splits in `dataset/train/`, `dataset/val/`, `dataset/test/`

**Step 04**

Mask RCNN training, more in `TRAINING_REPORT.md`
Output: `models/maskrcnn_final.pth`, `models/loss_curves.png`
For faster training, use Google Colab with GPU:
1. Upload `dataset/` and `models/train.py` to Google Drive
2. Mount Drive in Colab
3. Run `python models/train.py`
4. Download `maskrcnn_final.pth` back to local `models/` folder

**Step 05**

Evaluate model/
Output: Evaluation metrics printed and saved to `models/evaluation_results.json`

**Step 06**
Place envelope and ID card together in frame
Output: Measured images with dimensions in `measurement/outputs/`


### Repository Structure
```
xis_assessment/
├── calibration/
│ ├── Images/ # checkerboard photos
│ ├── calibrate.py
│ ├── undistort.py
│ ├── undistort_dataset.py
│ └── calibration_params.npz
│
├── dataset/
│ ├── Raw/ # original envelope photos
│ ├── undistorted/ # undistorted photos
│ ├── annotations/ # COCO JSON annotations
│ ├── train/
│ ├── val/
│ ├── test/
│ └── dataset_setup.py
│
├── models/
│ ├── train.py
│ ├── evaluate.py
│ ├── plot_loss_curve.py
│ ├── maskrcnn_final.pth
│ ├── loss_curves.png
│ └── evaluation_results.json
│
├── measurement/
│ ├── inputs/
│ ├── outputs/
│ └── measurement.py
│
├── docs/
│ ├── CALIBRATION_REPORT.md
│ ├── DATASET_CARD.md
│ ├── TRAINING_REPORT.md
│ ├── MEASUREMENT_REPORT.md
│ └── SETUP.md
│
├── requirements.txt
└── README.md
```