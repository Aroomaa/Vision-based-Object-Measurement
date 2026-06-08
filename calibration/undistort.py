import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

base_dir = os.getcwd()
calibration_file = os.path.join(base_dir, 'calibration', 'calibration_params.npz')
test_img_path = os.path.join(base_dir, 'calibration', 'Images')
output_path = os.path.join(base_dir, 'calibration', 'undistorted_sample.jpg')

data = np.load(calibration_file)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']
reprojection_error = data['reprojection_error']

print(f'Reprojection error: {reprojection_error:4f} px')
print(f'\nCamera matrix: \n{camera_matrix}')
print(f'\nDistortion coefficients: \n{dist_coeffs}')

all_files = os.listdir(test_img_path)
images = []
for file in all_files:
    if file.endswith('.jpg'):
        images.append(file)
if len(images) == 0:
    print('Error! Images not found.')
    exit()

test_img = os.path.join(test_img_path, images[14])

img = cv2.imread(test_img)
if img is None:
    print('Erro! Could not read the image.')
    exit()

img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, None)

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb_img_undistorted = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)

comparison_path = os.path.join(base_dir, 'calibration', 'undistortion_comparison.jpg')

plt.figure(figsize=(14,6))
plt.subplot(121); plt.imshow(rgb_img); plt.title('Original Distorted Image', fontsize = 14); plt.axis('off')
plt.subplot(122); plt.imshow(rgb_img_undistorted); plt.title('Undistorted Image', fontsize = 14); plt.axis('off')
plt.suptitle('Camera Callibration - Befor vs After', fontsize=16)
plt.tight_layout()
plt.savefig(comparison_path, dpi = 150, bbox_inches = 'tight')
plt.show()
print('Comparison image saved.')

cv2.imwrite(output_path, img_undistorted)
print('Undistorted image example saved.')