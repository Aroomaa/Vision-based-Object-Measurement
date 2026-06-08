import cv2
import numpy as np
import os
import glob

#configuration
base_dir = os.getcwd()
images_path = os.path.join(base_dir, 'calibration', 'Images', '*.jpg')
output_path = os.path.join(base_dir, 'calibration', 'calibration_params.npz')

checkerboard = (9, 7)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_matrix = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
obj_matrix[:,:2] = np.mgrid[0: checkerboard[0], 0: checkerboard[1]].T.reshape(-1, 2)

obj_points = []
img_points = []

images = glob.glob(images_path)
print(f'Total calibration images found are: {len(images)}')

if len(images) == 0:
    print('Error! No images found.')
    exit()

corner_imgs = 0

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f'Could not read {fname}')
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img_gray, checkerboard, None)
    
    if ret:
        corner_imgs += 1
        obj_points.append(obj_matrix)

        corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_refined)

        img_resized = cv2.resize(img, (800, 600))
        cv2.drawChessboardCorners(img_resized, checkerboard, corners_refined, ret)
        cv2.imshow('Checkerboard Corners', img_resized)
        cv2.waitKey(300)

        print(f"[{i+1}/{len(images)}] Corners found: {fname}")
    else:
        print(f"[{i+1}/{len(images)}] Corners not found: {fname}")

cv2.destroyAllWindows()
print(f'Total successful detections: {corner_imgs}/{len(images)}')


ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_gray.shape[::-1], None, None)

mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error

reprojection_error = mean_error / len(obj_points)
print(f'Reprojection error: {reprojection_error:.4f} px')

if reprojection_error < 0.3:
    print('Great calibration! (< 0.3 px)')
elif reprojection_error < 0.5:
    print('Good calibration! (< 0.5 px)')
else:
    print('Poor calibration. Recapture the images again.')

print(f'\nCamera matrix: \n{camera_matrix}')
print(f'\nDistortion coefficients: \n{dist_coeffs}')

np.savez(output_path, camera_matrix = camera_matrix, dist_coeffs = dist_coeffs, reprojection_error = reprojection_error)
print('Callibration parameters saved.')