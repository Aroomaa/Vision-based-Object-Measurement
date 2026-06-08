import os
import json
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

base_dir = os.getcwd()
model_path = os.path.join(base_dir, 'models', 'maskrcnn_final.pth')
calibration_file = os.path.join(base_dir, 'calibrataion', 'calibration_params.npz')
test_img_path = os.path.join(base_dir, 'dataset', 'test', 'images')
test_ann_path = os.path.join(base_dir, 'dataset', 'test', 'annotations', 'annotations.json')
device = torch.device('cpu')

num_classes = 2
confidence = 0.7

model = maskrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded.")

with open(test_ann_path, 'r') as f:
    coco_data = json.load(f)
test_images = coco_data['images']
test_annotations = coco_data['annotations']

coco_predictions = []
iou_scores = []
tp, fp, fn = 0, 0 ,0

for img_info in test_images:
    filename = os.path.basename(img_info['file_name'])
    img_path = os.path.join(test_img_path, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)
    
    scores = preds[0]['scores'].cpu().numpy()

    print(f"All scores: {scores[:5]}")  # show first 5 scores
    print(f"Max score: {max(scores) if len(scores) > 0 else 'none'}")

    boxes = preds[0]['boxes'].cpu().numpy()

    keep = scores >= confidence
    scores = scores[keep]
    boxes = boxes[keep]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        coco_predictions.append({'image_id': img_info['id'], 'category_id': 1, 'bbox': [x1, y1, x2-x1, y2-y1], 'score': float(scores[i])})

    ground_truth = []
    for a in test_annotations:
        if a['image_id'] == img_info['id']:
            ground_truth.append(a)

    
    if len(ground_truth) > 0 and len(boxes) > 0:
        gx, gy, gw, gh = ground_truth[0]['bbox']
        gx2, gy2 = gx+gw, gy+gh
        px1, py1, px2, py2 = boxes[0]

        ix1 = max(gx, px1)
        iy1 = max(gy, py1)
        ix2 = min(gx2, px2)
        iy2 = min(gy2, py2)
        inter = max(0, ix2-ix1)*max(0, iy2-iy1) 
        union = gw*gh + (px2-px1)*(py2-py1) - inter
        iou_scores.append(inter/union if union > 0 else 0)
        tp += 1
    
    elif len(ground_truth) > 0 and len(boxes) == 0:
        fn += 1
    elif len(ground_truth) == 0 and len(boxes) > 0:
        fp += 1

    print(f'{filename}  Detection: {len(boxes)}')

coco_groundtruth = COCO(test_ann_path)
coco_detection = coco_groundtruth.loadRes(coco_predictions)
coco_eval = COCOeval(coco_groundtruth, coco_detection, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()



map = coco_eval.stats[1]
map50_95 = coco_eval.stats[0]
mean_iou = np.mean(iou_scores) if iou_scores else 0
precision = tp/(tp+fp) if (tp+fp) > 0 else 0
recall = tp/(tp+fn) if (tp+fn) > 0 else 0
f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

print(f"mAP@0.5:      {map:.4f}")
print(f"mAP@0.5:0.95: {map50_95:.4f}")
print(f"Mean IoU:     {mean_iou:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1 Score:     {f1:.4f}")

results = results = {'mAP@0.5': round(float(map), 4), 'mAP@0.5:0.95': round(float(map50_95), 4), 'Mean IoU': round(float(mean_iou), 4),
    'Precision': round(float(precision), 4), 'Recall': round(float(recall), 4), 'F1': round(float(f1), 4)}

with open(os.path.join(base_dir, 'models', 'evaluation_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

print('Results saved.')