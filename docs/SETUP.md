# Setup and Installation Guide

### System Requirements

| **Component** | **Requirement** |
|---------------|---------------- |
| Python | 3.9 or higher |
| RAM | 8 GB minimum (16 GB recommended for training) |
| GPU | Optional  |
| Disk space | ~2 GB (images + model weights)

### Installation
```
git clone < >
cd < >
```

### Install Dependencies

`pip install -r requirements.txt`

### Directory Preparation

```
mkdir -p calibration/Image
mkdir -p dataset/Raw
mkdir -p docs
mkdir -p inference
mkdir -p models
mkdir -p measurement
```

### Procedure

**Step 01**

Camera calibration
Print corner detection results
Print camera matrix and distortion coefficients
Print reprojection error
Save calibration parameters

**Step 02**

Verfy undistortion
Comapre distortion and undistortion results

**Step 03**

Save undistort dataset

**Step 04**

Get aanotation `.json` file
Split daatset into train, validation, test

**Step 05**

Mask RCNN training, more in `TRAINING_REPORT.md`

**Step 06**

Run inference and measure the test object in mm, more in `MEASUREMENT_REPORT.md`
