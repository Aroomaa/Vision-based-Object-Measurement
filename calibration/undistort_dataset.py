import cv2
import numpy as np
import os

base_dir = os.getcwd()
calibration_file = os.path.join(base_dir, 'calibration', 'calibration_params.npz')
input_folder = os.path.join(base_dir, 'dataset', 'Raw')
output_folder = os.path.join(base_dir, 'dataset', 'undistorted')

data = np.load(calibration_file)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

all_files = os.listdir(input_folder)
images = []
for file in all_files:
    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
        images.append(file)
print(f'Total images found: {len(images)}')
if len(images) == 0:
    print('Error! Images not found.')
    exit()

undistorted_imgs = 0
failed_imgs = 0

for i, filename in enumerate(images):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    img = cv2.imread(input_path)
    if img is None:
        print(f"[{i+1}/{len(images)}] Could not read: {filename}")
        failed_imgs += 1

    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, None)

    cv2.imwrite(output_path, img_undistorted)
    undistorted_imgs += 1
    print(f"[{i+1}/{len(images)}] Undistorted: {filename}")
    
print('\nUndistortion complete.')

print(f'Successfully Undistorted Images: {undistorted_imgs}/{len(images)}')
print(f'\nFailed Images: {failed_imgs}/{len(images)}')