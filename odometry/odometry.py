import numpy as np
import cv2
from matplotlib import pyplot as plt
from odometry.m6bk import *
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)


def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    #surf = cv2.xfeatures2d.SURF_create(400)
    #kp, des = surf.detectAndCompute(image,None)
    orb = cv2.ORB_create(nfeatures=1500)
    #orb = cv2.xfeatures2d.ORB_create(400)
    kp, des = orb.detectAndCompute(image,None)
    return kp, des

def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plt.imshow(display)

def extract_features_dataset(images):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images

    """
    kp_list = []
    des_list = []

    for i in range(len(images)):
        kp, des = extract_features(images[i])
        kp_list+=[kp]
        des_list+=[des]
    return kp_list, des_list


def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    des1=np.array(des1)
    des2=np.array(des2)

    matches = bf.match(des1, des2)

    match = sorted(matches, key = lambda x:x.distance)
    return match

def match_features_dataset(des_list):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset.
               Each matches[i] is a list of matched features from images i and i + 1

    """
    matches = []

    i = 0
    for i in range(len(des_list) - 1):
        des1 = des_list[i]
        des2 = des_list[i+1]
        match = match_features(des1, des2)
        matches.append(match)

    return matches

def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    return match
    '''
    filtered_match = []

    for i in range(len(match)):
        if match[i][0]>dist_threshold:
            filtered_match+=[match[i]]
    '''
    return filtered_match

def filter_matches_dataset(matches, dist_threshold=0.6):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    matches -- list of matches for each subsequent image pair in the dataset.
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset.
                        Each matches[i] is a list of good matches, satisfying the distance threshold
    """
    filtered_matches = []
    for match in matches:
        filtered_match = filter_matches_distance(match, dist_threshold)
        filtered_matches.append(filtered_match)

    return filtered_matches

def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


def estimate_trajectory(matches, kp_list, k, depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    matches -- list of matches for each subsequent image pair in the dataset.
               Each matches[i] is a list of matched features from images i and i + 1
    des_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix

    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and
                  trajectory[:, i] is a 3x1 numpy vector, such as:
                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location

                  * Consider that the origin of your trajectory cordinate system is located at the camera position
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven
                  at the initialization of this function

    """
    def estimate_motion(match, kp1, kp2, k, depth1=None):
        """
        Estimate camera motion from a pair of subsequent image frames

        Arguments:
        match -- list of matched features from the pair of images
        kp1 -- list of the keypoints in the first image
        kp2 -- list of the keypoints in the second image
        k -- camera calibration matrix

        Optional arguments:
        depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

        Returns:
        rmat -- recovered 3x3 rotation numpy matrix
        tvec -- recovered 3x1 translation numpy vector
        image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system
        image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system

        """
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        image1_points = []
        image2_points = []

        image1_points = [kp1[m.queryIdx].pt for m in match]
        image2_points = [kp2[m.trainIdx].pt for m in match]
        i1_pts = np.array(image1_points)
        i2_pts = np.array(image2_points)
        E, _ = cv2.findEssentialMat(i1_pts, i2_pts, k)
        _, rmat, tvec, _ = cv2.recoverPose(E, i1_pts, i2_pts, k)

        return rmat, tvec, image1_points, image2_points


    trajectory = np.zeros((3, 1))

    trajectory = [np.array([0, 0, 0])]
    P = np.eye(4)

    for i in range(len(matches)):
        match = matches[i]
        kp1 = kp_list[i]
        kp2 = kp_list[i+1]
        depth =  depth_maps[i] if len(depth_maps)>0 else None

        rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth)
        R = rmat
        t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])

        P_new = np.eye(4)
        P_new[0:3,0:3] = R.T
        P_new[0:3,3] = (-R.T).dot(t)
        P = P.dot(P_new)

        trajectory.append(P[:3,3])

    trajectory = np.array(trajectory).T
    trajectory[2,:] = -1*trajectory[2,:]

    return trajectory

def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    image1 = image1.copy()
    image2 = image2.copy()

    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv2.circle(image1, p1, 5, (0, 255, 0), 1)
        cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv2.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv2.circle(image2, p2, 5, (255, 0, 0), 1)

    if is_show_img_after_move:
        return image2
    else:
        return image1


def visualize_trajectory(trajectory):
    # Unpack X Y Z each trajectory point
    locX = []
    locY = []
    locZ = []
    # This values are required for keeping equal scale on each plot.
    # matplotlib equal axis may be somewhat confusing in some situations because of its various scale on
    # different axis on multiple plots
    max = -math.inf
    min = math.inf

    # Needed for better visualisation
    maxY = -math.inf
    minY = math.inf

    for i in range(0, trajectory.shape[1]):
        current_pos = trajectory[:, i]

        locX.append(current_pos.item(0))
        locY.append(current_pos.item(1))
        locZ.append(current_pos.item(2))
        if np.amax(current_pos) > max:
            max = np.amax(current_pos)
        if np.amin(current_pos) < min:
            min = np.amin(current_pos)

        if current_pos.item(1) > maxY:
            maxY = current_pos.item(1)
        if current_pos.item(1) < minY:
            minY = current_pos.item(1)

    auxY_line = locY[0] + locY[-1]
    if max > 0 and min > 0:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2
    elif max < 0 and min < 0:
        minY = auxY_line + (min - max) / 2
        maxY = auxY_line - (min - max) / 2
    else:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 3)
    ZY_plt = plt.subplot(gspec[0, 1:])
    YX_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    # Actual trajectory plotting ZX
    toffset = 1.06
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    traj_main_plt.plot(locZ, locX, ".-", label="Trajectory", zorder=1, linewidth=1, markersize=4)
    traj_main_plt.set_xlabel("Z")
    # traj_main_plt.axes.yaxis.set_ticklabels([])
    # Plot reference lines
    traj_main_plt.plot([locZ[0], locZ[-1]], [locX[0], locX[-1]], "--", label="Auxiliary line", zorder=0, linewidth=1)
    # Plot camera initial location
    traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    traj_main_plt.set_xlim([min, max])
    traj_main_plt.set_ylim([min, max])
    traj_main_plt.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Plot ZY
    # ZY_plt.set_title("Z Y", y=toffset)
    ZY_plt.set_ylabel("Y", labelpad=-4)
    ZY_plt.axes.xaxis.set_ticklabels([])
    ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
    ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    ZY_plt.set_xlim([min, max])
    ZY_plt.set_ylim([minY, maxY])

    # Plot YX
    # YX_plt.set_title("Y X", y=toffset)
    YX_plt.set_ylabel("X")
    YX_plt.set_xlabel("Y")
    YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
    YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
    YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([min, max])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=toffset)
    D3_plt.plot3D(locX, locZ, locY, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(min, max)
    D3_plt.set_ylim3d(min, max)
    D3_plt.set_zlim3d(min, max)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Z", labelpad=0)
    D3_plt.set_zlabel("Y", labelpad=-2)

    # plt.axis('equal')
    D3_plt.view_init(45, azim=30)
    plt.tight_layout()
    plt.show()
