import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

base_dir = os.getcwd()
model_path = os.path.join(base_dir, 'models', 'maskrcnn_final.pth')
calibration_file = os.path.join(base_dir, 'calibration', 'calibration_params.npz')
output_dir = os.path.join(base_dir, 'measurement', 'outputs')
os.makedirs(output_dir, exist_ok=True)

envelope_real_width_mm  = 90.0
envelope_real_height_mm = 139.0
num_classes = 2
confidence = 0.7
device = torch.device('cpu')

calibration = np.load(calibration_file)
camera_matrix = calibration['camera_matrix']
dist_coeffs   = calibration['dist_coeffs']

model = maskrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, num_classes)
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
model.load_state_dict(torch.load(model_path, map_location = device))
model.to(device)
model.eval()
print("Model loaded!")

