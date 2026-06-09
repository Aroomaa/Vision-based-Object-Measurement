import os
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

base_dir = os.getcwd()

train_img_dir = os.path.join(base_dir, "dataset/train/images")
train_ann = os.path.join(base_dir, "dataset/train/annotations/annotations.json")
val_img_dir = os.path.join(base_dir, "dataset/val/images")
val_ann = os.path.join(base_dir, "dataset/val/annotations/annotations.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

num_classes = 2
num_epochs = 50
batch_size = 2
learning_rate = 0.005

class cocoDataset(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = F.to_tensor(img)

        boxes = []
        labels = []
        masks = []

        for obj in target:
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x+w, y+h])
            labels.append(1)

            mask = np.zeros((img.shape[1], img.shape[2]), dtype = np.uint8)
            for seg in obj['segmentation']:
                pts = np.array(seg, dtype = np.inst32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

        boxes = torch.as_tensor (boxes, dtype = torch.float32)
        labels = torch.as_tensor (labels, dtype = torch.int64)
        masks = torch.as_tensor (np.array(masks), dtype = torch.uint8)

        target_dict = {'boxes': boxes, 'labels': labels, 'masks': masks, 'image_id': torch.tensor([idx])} 
        return img, target_dict
    
train_dataset = cocoDataset(train_img_dir, train_ann)
val_dataset = cocoDataset(val_img_dir, val_ann)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())
                   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.train()

    total_val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss      = sum(l for l in loss_dict.values())
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    lr_scheduler.step()
    print(f'Epoch [{epoch+1} / {num_epochs}] Train loss: {avg_train_loss:.4f} & Validation loss: {avg_val_loss:4f}')

torch.save(model.state.dict(), os.path.join(base_dir, 'models', 'maskrcnn_final.pth'))
print('\nFinal model saved.')

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs +1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs +1), val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mask R-CNN — Loss Curves')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'models', 'loss_curves.png'))
plt.show()
print('Loss curves saved.')
print('Training complete!')