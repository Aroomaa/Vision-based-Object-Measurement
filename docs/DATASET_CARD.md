# Dataset Information

### Object Description

**Object:** Envelope

**Material:** Hard card paper

**Dimensions:** 139 x 90 mm

**Task:** Instance segmentation

**Total Images:** 100

**Total Annotations:** 100


### **Why I Chose this Object**

I chose hard paper envelop becuase
- It's rigid rectangular shape makes it suitable for segmentation and measurement tasks.
- It's edges are clearly defined which makes it better for annotation.
- It's flat structure reduces distortion during image capturing.
- Any hard paper structure like this is commonly available.

### **Collection Methodology**

The images are taken using smartphone camera under different light conditions and angles.
1. Camera was first gully calibrated (see `CALIBRATION_REPORT.md`)
2. All images are captured the same calibrated camera to ensure consistency.
3. All images are undistorted using `undistort_dataset.py` before labelling and splitting.

**Backgrounds:** Plain black, plain white and cluttered backgrounds 

**Lighting:** Natural daylight and indoor artificial light

**Angles:** Front angle, tilted angle and slight rotations 

**Distances:** Close up shots and medium range shots


### **Dataset**

**Size:** Total images collected are 100, enough to meet minimum requirement for training segmentation model

### **Annotation Methodology**

Each image is manually labelled to outline envelop boundaries

**Tool Used:** CVAT

**Annotation Type:** Polygon segmentation masks

**Annotation Format:** COCO JSON(polygon masks)


### **Classes**

**Class 1:** Envelope


### **Dataset Splitting**

The dataset is divided into following sets to ensure proper model training, hyperparameter tuning and accurate performance evaluation on unseen data

**Training set:** 70%

**Validation Set:** 20%

**Test Set:** 10%

Splitting is performed on full image list after random shuffling. Each split contains:
- An    images/` folder with undistorted PG files.
- An `annotations/annotations.json` COCO-format file containing only annotations belonging to that split's image IDs.

### Directory Structure 

```
dataset/
├── Raw/                        # Original captured images (pre-undistortion)
├── undistorted/               # Undistorted images (used for labelling & training)
├── annotations/
│   └── instances_default.json # Full COCO annotation file from CVAT
├── train/
│   ├── images/
│   └── annotations/annotations.json
├── val/
│   ├── images/
│   └── annotations/annotations.json
└── test/
    ├── images/
    └── annotations/annotations.json
```

