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
input_dir = os.path.join(base_dir, 'measurement', 'inputs')
os.makedirs(output_dir, exist_ok=True)

ref_longside_mm = 85.6
ref_shortside_mm = 54.0
real_width_mm = 90.0
real_height_mm = 139.0
num_classes = 2
confidence = 0.7
device = torch.device('cpu')

hsv_l1 = np.array([0,   88,  42],  dtype=np.uint8)
hsv_h1 = np.array([10,  255, 207], dtype=np.uint8)
hsv_l2 = np.array([165, 88,  42],  dtype=np.uint8)
hsv_h2 = np.array([180, 255, 207], dtype=np.uint8)

calibration = np.load(calibration_file)
camera_matrix = calibration['camera_matrix']
dist_coeffs = calibration['dist_coeffs']

model = maskrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded.")

def detect_ref_card(img):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, hsv_l1, hsv_h1), cv2.inRange(hsv, hsv_l2, hsv_h2))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    img_area = img.shape[0] * img.shape[1]
    valid_ref_det = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > img_area * 0.50:
            continue

        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        long_side = max(w, h)
        px_per_mm = long_side / ref_longside_mm

        if px_per_mm < 3 or px_per_mm > 100:
            continue

        valid_ref_det.append((px_per_mm, area))

    if not valid_ref_det:
        return None

    valid_ref_det.sort(key=lambda x: x[1], reverse=True)
    return valid_ref_det[0][0]


def measure(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f'Error! Could not read image.')
        return None

    img = cv2.undistort(img, camera_matrix, dist_coeffs, None, None)

    pixels_per_mm = detect_ref_card(img)
    if pixels_per_mm is None:
        print('Reference card not detected.')
        return None
    print(f"Pixels per mm: {pixels_per_mm:.4f}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)

    scores = preds[0]['scores'].cpu().numpy()
    boxes = preds[0]['boxes'].cpu().numpy()
    masks = preds[0]['masks'].cpu().numpy()

    keep = scores >= confidenscores = scores[keep]
    boxes = boxes[keep]
    masks = masks[keep]

    if len(scores) == 0:
        print("No envelope detected!")
        return None

    best = np.argmax(scores)
    mask_binary = (masks[best, 0] > 0.5).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('No contour found.')
        return None

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    pts = np.int32(cv2.boxPoints(rect))
    side1, side2 = rect[1]

    height_px = max(side1, side2)
    width_px  = min(side1, side2)
    width_mm  = width_px / pixels_per_mm
    height_mm = height_px / pixels_per_mm
    error_width = abs(width_mm - real_width_mm)
    error_height = abs(height_mm - real_height_mm)

    print('\nMEASUREMENT RESULTS ')
    print(f'Confidence : {scores[best]:.4f}')
    print(f'Width : {width_px:.1f} px & {width_mm:.2f} mm  (actual {real_width_mm}mm, err {error_width:.2f}mm)')
    print(f'Height : {height_px:.1f} px & {height_mm:.2f} mm  (actual {real_height_mm}mm, {error_height:.2f}mm)\n')
    n
    img_display = img.copy()

    # Width midpoints blue line
    mid_w1 = ((pts[0][0] + pts[3][0]) // 2, (pts[0][1] + pts[3][1]) // 2)
    mid_w2 = ((pts[1][0] + pts[2][0]) // 2, (pts[1][1] + pts[2][1]) // 2)

    # Height midpoints cyan line
    mid_h1 = ((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2)
    mid_h2 = ((pts[2][0] + pts[3][0]) // 2, (pts[2][1] + pts[3][1]) // 2)

    cv2.line(img_display, mid_w1, mid_w2, (255, 80,  0),  3)
    cv2.line(img_display, mid_h1, mid_h2, (0,  220, 255), 3)

    tx = int(min(pts[:, 0])) + 10
    ty = max(int(min(pts[:, 1])) - 20, 90)

    cv2.putText(img_display, f'W: {width_mm:.1f}mm', (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 80, 0), 3)
    cv2.putText(img_display, f'H: {height_mm:.1f}mm', (tx, ty+55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3)
    cv2.putText(img_display, f'Conf: {scores[best]:.2f}', (tx, ty+110), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title(f'W: {width_mm:.1f}mm & H: {height_mm:.1f}mm - Conf: {scores[best]:.2f}')
    plt.axis('off')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'measured_{os.path.basename(image_path)}')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved')

    return width_mm, height_mm, float(scores[best])


# Run on all input images
image_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
results_list = []

print(f'Found images: {len(image_files)}')

for fname in image_files:
    result = measure(os.path.join(input_dir, fname))
    if result is None:
        continue
    w, h, conf = result
    results_list.append({'filename': fname, 'width': w, 'height': h, 'confidence': conf})

if not results_list:
    print('\nNo valid measurements obtained.')
else:
    widths  = np.array([r['width']  for r in results_list])
    heights = np.array([r['height'] for r in results_list])

    print(f"\n{'='*47}")
    print(f'Total processed : {len(results_list)}')
    print(f'Mean Width : {np.mean(widths):.2f} mm  (actual {real_width_mm}mm)')
    print(f'Mean Height : {np.mean(heights):.2f} mm  (actual {real_height_mm}mm)')
    print(f'Mean Absolute Error Width  : {np.mean(np.abs(widths  - real_width_mm)):.2f} mm')
    print(f'Mean Absolute Error Height =: {np.mean(np.abs(heights - real_height_mm)):.2f} mm')
    print(f'Mean Percentage Error Width : {np.mean(np.abs(widths  - real_width_mm)  / real_width_mm  * 100):.2f}%')
    print(f'Mean Percentage Error Height : {np.mean(np.abs(heights - real_height_mm) / real_height_mm * 100):.2f}%')