import numpy as np
import cv2
from matplotlib import pyplot as plt
from odometry.m6bk import *
from odometry.consts import *
np.random.seed(1)

def xy_from_depth(depth, k):
    """
    Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.

    Arguments:
    depth -- tensor of dimension (H, W), contains a depth value (in meters) for every pixel in the image.
    k -- tensor of dimension (3x3), the intrinsic camera matrix

    Returns:
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    """
    f = k[0,0]
    uc = k[0,2]
    uv = k[1,2]
    shape=depth.shape
    # Get the shape of the depth tensor

    # Grab required parameters from the K matrix

    # Generate a grid of coordinates corresponding to the shape of the depth map
    x = np.copy(depth)
    y = np.copy(depth)
    # Compute x and y coordinates
    for h in range(shape[0]):
        for w in range(shape[1]):
            x[h,w] = ((w+1-uc)*depth[h,w])/f
            y[h,w] = ((h+1-uv)*depth[h,w])/f
    return x, y


def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr --
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """

    num_itr = 100  # RANSAC maximum number of iterations
    N=xyz_data.shape[1]
    min_num_inliners = int(N - 0.01*N)  # RANSAC minimum number of inliers
    distance_threshold = 0.001 #TODO(res)  # Maximum distance from point to plane for point to be considered inlier
    num_inliners=0
    gxyz=None
    xr = xyz_data[0,:]
    yr = xyz_data[1,:]
    zr = xyz_data[2,:]
    for i in range(num_itr):
        #at least 3 non-colinear points, to be save 15 random points are sampled
        rs_idx = np.random.choice(N, 15, replace=False)
        plane = compute_plane(xyz_data[:,rs_idx])
        dst = np.abs(dist_to_plane(plane, xr, yr, zr))
        cur_num_inliners = sum(dst < distance_threshold)
        if cur_num_inliners > num_inliners:
            num_inliners=cur_num_inliners
            gxyz=xyz_data[:,dst<distance_threshold]
        if num_inliners>min_num_inliners:
            break
    return compute_plane(gxyz)

def estimate_lane_lines(segmentation_output):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    segmentation_output -- tensor of dimension (H,W), containing semantic segmentation neural network output
    minLineLength -- Scalar, the minimum line length
    maxLineGap -- Scalar, dimension (Nx1), containing the z coordinates of the points

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """
    lane_and_road = np.zeros(segmentation_output.shape)
    lane_and_road[segmentation_output==ROAD_FLAG] = 255
    lane_and_road[segmentation_output==LANE_FLAG] = 255
    lane_and_road=lane_and_road.astype(np.uint8)
    #
    lane_and_road=cv2.GaussianBlur(lane_and_road, (5,5), 1)
    edgy = cv2.Canny(lane_and_road, 10, 110) #[100,110] now [10-110]
    #
    lines = cv2.HoughLinesP(edgy, 3, np.pi/180, threshold=100, minLineLength=100, maxLineGap=40) #[]100, 25] now [3, np.pi/180, 100,100, 40]
    lines = np.squeeze(lines, axis=1)
    return lines

def merge_lane_lines(lines):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """
    # Step 0: Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3

    slopes, intercepts = get_slope_intercept(lines)
    slopes_filter=[]
    intercepts_filter=[]
    # illuminate horizontal lines
    for i in range(len(slopes)):
        if abs(slopes[i]) > min_slope_threshold:
            slopes_filter+=[slopes[i]]
            intercepts_filter+=[intercepts[i]]
    slopes=slopes_filter
    intercepts=intercepts_filter
    # Cluster lines with similarity
    # Cluster list, list of  clusters, where each cluster is a list of close lines
    C = []
    for slope, intercept in zip(slopes, intercepts):
        cluster = [(slope, intercept)]
        for s, i in zip (slopes, intercepts):
            if abs(slope-s)<slope_similarity_threshold and abs(intercept-i)<intercept_similarity_threshold:
                cluster+=[(s,i)]
        C+=[cluster]
    # average clusters
    merged_lines =[]
    for c in C:
        L = len(c)
        S = sum(np.array(c))
        avg = S/L # the average line (slope, intercept)
        #TODO (fix) use lines instead of slopes/intercepts in filtering
        x2=4000 # Cluster list, list of  clusters, where each cluster is a list of close lines
        line = np.array([0,avg[1], x2, avg[0]*x2+avg[1]])
        merged_lines+=[line]
    return np.array(merged_lines)

def filter_detections_by_segmentation(detections, segmentation_output):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    segmentation_output -- tensor of dimension (HxW) containing pixel category labels.

    Returns:
    filtered_detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    """
    ratio_threshold = 0.3  # If 1/3 of the total pixels belong to the target category, the detection is correct.
    filtered_detections = []
    for detection in detections:
            cls, *nums = detection
            x1, y1, x2, y2, score = np.asfarray(nums)
            x1_idx,x2_idx,y1_idx,y2_idx=int(y1),int(y2),int(x1),int(x2)
            cls_idx=None
            if cls==CAR_CLS or cls==MOTORBIKE_CLS or cls==BUS_CLS or cls==TRAIN_CLS or cls==TRUCK_CLS:
                cls_idx=VEHICLE_FLAG
            elif cls==PEDESTRIAN_CLS:
                cls_idx=PEDESTRIAN_FLAG
            else:
                continue
            area = (y2-y1)*(x2-x1)
            canvas = np.zeros(segmentation_output.shape)
            cropped_canvas = canvas[x1_idx:x2_idx,y1_idx:y2_idx]
            cropped_segmentation=segmentation_output[x1_idx:x2_idx,y1_idx:y2_idx]
            cropped_canvas[cropped_segmentation==cls_idx] = 1

            TP = sum(sum(cropped_canvas))
            iou =  TP/area
            #print('TP: {}, area: {}, iou: {}'.format(TP, area, iou))
            if iou>0.3:
                filtered_detections+=[detection]
    return filtered_detections


def find_min_distance_to_detection(detections, x, y, z):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    z -- tensor of dimensions (H,W) containing the z coordinates of every pixel in the camera coordinate frame.
    Returns:
    min_distances -- tensor of dimension (N, 1) containing distance to impact with every object in the scene.

    """
    min_distances=[]
    for detection in detections:
        shape=x.shape
        distance=np.zeros(shape)
        for h in range(shape[0]):
            for w in range(shape[1]):
                distance[h,w] = np.sqrt(x[h,w]**2+y[h,w]**2+z[h,w]**2)
        cls, *nums = detection
        x1, y1, x2, y2, score = np.asfarray(nums)
        x1_idx,x2_idx,y1_idx,y2_idx=int(y1),int(y2),int(x1),int(x2)
        cropped_distance=distance[x1_idx:x2_idx,y1_idx:y2_idx]
        #print('detection: {}, detection distance: {}'.format(detection, cropped_distance))
        min_dist=np.min(cropped_distance)
        #print('min_dist: {}'.format(min_dist))
        #assert(min_dist-0.1>0)
        min_distances+=[min_dist]
    return min_distances


def vis_object_detection(image_out, objects, min_distance):

    colour_scheme = {'Car': (255, 255, 102),
                     'Cyclist': (102, 255, 255),
                     'Pedestrian': (255, 102, 255),
                     'Background': (0, 0, 255)}

    for obj in objects:
        category = obj[0]
        #TODO (fix)
        if category==PEDESTRIAN_CLS:
            category='Pedestrian'
        elif category==BICYCLE_CLS:
            category='Cyclist'
        elif category==CAR_CLS:
            category='Car'
        else:
            #assert(False, 'you shouldnt get his obj class: {}'.format(category))
            category='Background'

        bounding_box = np.asfarray(obj[1:5]).astype(int)

        image_out = cv2.rectangle(image_out.astype(np.uint8),
                                  (bounding_box[0], bounding_box[1]),
                                  (bounding_box[2], bounding_box[3]),
                                  colour_scheme[category],
                                  4)

    for detection, min_distance in zip(objects, min_distance):
        bounding_box = np.asfarray(detection[1:5])
        text =  str(np.round(min_distance, 2)) + ' m'
        bl = (int(bounding_box[0]), int(bounding_box[1]-20))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_out, text, bl, font, 1, (255,27,169), 1)
        #plt.text(bounding_box[0], bounding_box[1] - 20, text, fontdict=font)

    return image_out

def vis_lanes(image_out, lane_lines):
    for line in lane_lines:
        x1, y1, x2, y2 = line.astype(int)

        image_out = cv2.line(
            image_out.astype(
                np.uint8), (x1, y1), (x2, y2), (255, 0, 255), 7)

    return image_out
