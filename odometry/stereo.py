import numpy as np
import cv2
from matplotlib import patches

def compute_left_disparity_map(img_left, img_right):
    # Parameters
    num_disparities = 6*16
    block_size = 11

    min_disparity = 0
    window_size = 6

    #TODO (fix) Expected Ptr<cv::UMat> for argument '%s'
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    # Stereo BM matcher
    left_matcher_BM = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )
    # Stereo SGBM matcher
    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    # Compute the left disparity map
    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16

    return disp_left

def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = t / t[3]
    return k, r, t


def calc_depth_map(disp_left, k_left, t_left, t_right):

    # Get the focal length from the K matrix
    f = k_left[0, 0]

    # Get the distance between the cameras from the t matrices (baseline)
    b = t_left[1] - t_right[1]

    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disp_left.shape, np.single)

    # Calculate the depths
    depth_map[:] = f * b / disp_left[:]

    return depth_map



def locate_obstacle_in_image(image, obstacle_image):

    # Run the template matching from OpenCV
    cross_corr_map = cv2.matchTemplate(image, obstacle_image, method=cv2.TM_CCOEFF)

    # Locate the position of the obstacle using the minMaxLoc function from OpenCV
    _, _, _, obstacle_location = cv2.minMaxLoc(cross_corr_map)

    return cross_corr_map, obstacle_location

def calculate_nearest_point(depth_map, obstacle_location, obstacle_img):
    # Gather the relative parameters of the obstacle box
    obstacle_width = obstacle_img.shape[0]
    obstacle_height = obstacle_img.shape[1]
    obstacle_min_x_pos = obstacle_location[1]
    obstacle_max_x_pos = obstacle_location[1] + obstacle_width
    obstacle_min_y_pos = obstacle_location[0]
    obstacle_max_y_pos = obstacle_location[0] + obstacle_height
    # Get the depth of the pixels within the bounds of the obstacle image, find the closest point in this rectangle
    obstacle_depth = depth_map_left[obstacle_min_x_pos:obstacle_max_x_pos, obstacle_min_y_pos:obstacle_max_y_pos]
    closest_point_depth = obstacle_depth.min()
    # Create the obstacle bounding box
    obstacle_bbox = patches.Rectangle((obstacle_min_y_pos, obstacle_min_x_pos), obstacle_height, obstacle_width, linewidth=1, edgecolor='r', facecolor='none')
    return closest_point_depth, obstacle_bbox
