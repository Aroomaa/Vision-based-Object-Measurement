import os 
import json
import random
import shutil

base_dir = os.getcwd()
annotations_file = os.path.join(base_dir, 'dataset', 'annotations', 'instances_default.json')
images_path = os.path.join(base_dir, 'dataset', 'undistorted')
output_path = os.path.join(base_dir, 'dataset')

train_ratio = 0.70
val_ratio = 0.20
test_ratio = 0.10

with open(annotations_file, 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']
categories = coco_data['categories']

print(f'Total images: {len(images)}')
print(f'Total annotations: {len(annotations)}')
print(f"Categories: {[c['name'] for c in categories]}")

random.seed(42)
random.shuffle(images)

total = len(images)
train_end = int(total * train_ratio)
val_end = train_end +int(total* val_ratio)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

def create_split(split_images, split_name):
    img_folder = os.path.join(output_path, split_name, 'images')
    ann_folder = os.path.join(output_path, split_name, 'annotations')
    os.makedirs(img_folder, exist_ok = True)
    os.makedirs(ann_folder, exist_ok = True)
    ann_file = os.path.join(ann_folder,'annotations.json')

    image_ids = []
    for img in split_images:
        image_ids.append(img['id'])
    
    split_annotations = []
    for ann in annotations:
        if ann['image_id'] in image_ids:
            split_annotations.append(ann)

    split_coco = {'images': split_images, 'annotations': split_annotations, 'categories': categories}
    with open(ann_file, 'w') as f:
        json.dump(split_coco, f, indent = 4)
    
    copied = 0

    for img in split_images:
        filename = img['file_name']
        filename = os.path.basename(filename)
        src = os.path.join(images_path, filename)
        dst = os.path.join(img_folder, filename)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
        else:
            print(f'Error! File not found: {filename}')

create_split(train_images, 'train')
create_split(val_images, 'val')
create_split(test_images, 'test')

print('\nSplit Results')
print(f'Train images: {len(train_images)} \nValudation images: {len(val_images)} \nTest images: {len(test_images)}')

