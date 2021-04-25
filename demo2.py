from odometry.camera import FileCamera
from odometry.perception import *
from odometry.stereo import *
from odometry.m6bk import DatasetHandler
from odometry.odometry import *
from matplotlib import pyplot as plt
from segmentation.segment import *
from PIL import Image
import os

# the demo is build using stero imaging from:
# camer calibration parameters for mrpt/karlsruhe_dataset is taken from https://www.mrpt.org/Karlsruhe_Dataset_Rawlog_Format
# DIR is set to Karlsruhe dataset
DIR='/opt/stereo/2010_03_09_drive_0019/2010_03_09_drive_0019_Images'
#left camera
P1=np.array([6.452401e+02, 0.000000e+00, 6.359587e+02, 0.000000e+00, 0.000000e+00, 6.452401e+02, 1.941291e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00])
P1=P1.reshape(3,4)

left_cam = FileCamera(P1, DIR)
left_P = left_cam.get_projection_matrices()
k_left, _, t_left = decompose_projection_matrix(left_P)
#k_left= np.array([8.941981e+02, 0.000000e+00, 6.601406e+02, 0.000000e+00, 8.927151e+02, 2.611004e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape((3,3))
#
#right camera
P2=np.array([6.452401e+02, 0.000000e+00, 6.359587e+02, -3.682632e+02, 0.000000e+00, 6.452401e+02, 1.941291e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00])
P2=P2.reshape(3,4)

right_cam = FileCamera(P2, DIR)
right_P = right_cam.get_projection_matrices()
k_right, _, t_right = decompose_projection_matrix(right_P)
#k_right=np.array([8.800704e+02, 0.000000e+00, 6.635881e+02, 0.000000e+00, 8.798504e+02, 2.690108e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape((3,3))

L=left_cam.size()
data_handler=DatasetHandler()

images = []
for i in range(left_cam.size()):
    left, right=left_cam.get_image(i)
    images.append(left)

kp_list, des_list = extract_features_dataset(images)
matches = match_features_dataset(des_list)
#TODO (res)
filtered_matches = filter_matches_dataset(matches)
matches = filtered_matches
#
trajectory = estimate_trajectory(matches, kp_list, k_left)
visualize_trajectory(trajectory)
