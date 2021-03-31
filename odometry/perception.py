import numpy as np
import cv2
from matplotlib import pyplot as plt
from odometry.m6bk import *
from odometry.lane_det import *
from odometry.consts import *

def perceive(image, segmentation, depth, k, detections):
    #
    z = depth
    k = k
    x, y = xy_from_depth(z, k)
    print('Z: {}, X: {}, Y: {}'.format(np.sum(z), np.sum(x), np.sum(y)))
    #
    road_mask = np.zeros(segmentation.shape)
    road_mask[segmentation == ROAD_FLAG] = 1
    print('x shape: {}, segmentation shape: {}, road_mask shape: {}'.format(x.shape, segmentation.shape, road_mask.shape))
    x_ground = x[road_mask == 1]
    y_ground = y[road_mask == 1]
    z_ground = depth[road_mask == 1]
    print('x_ground shape: {}, y_ground shape: {}, z_ground shape: {}'.format(x_ground, y_ground, z_ground))
    xyz_ground = np.stack((x_ground, y_ground, z_ground))
    print('x_ground shape: {}, y_ground shape: {}, z_ground shape: {}, xyz shape: {}'.format(x_ground, y_ground, z_ground, xyz_ground))
    p_final = ransac_plane_fit(xyz_ground)
    #
    lane_lines = estimate_lane_lines(segmentation)
    print('lane lines: {}'.format(lane_lines))
    merged_lane_lines = merge_lane_lines(lane_lines)
    print('merged lane lines: {}'.format(merged_lane_lines))
    #
    max_y = image.shape[0]
    min_y = np.min(np.argwhere(road_mask == 1)[:, 0])
    #
    if len(merged_lane_lines)>0:
        extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
    else:
        extrapolated_lanes = []
    #mp=dataset_handler.lane_midpoint
    #TODO calculate lane area center from w.r.t camera aspect ratio
    print('extrapolated lanes: {}'.format(extrapolated_lanes))
    mp=[175,700]

    final_lanes = find_closest_lines(extrapolated_lanes, mp)
    #
    print('final lanes: {}'.format(final_lanes))
    #TODO (remove)
    if len(final_lanes)==0:
        if not len(extrapolated_lanes) == 0:
            final_lanes=extrapolated_lanes
        if len(final_lanes)==0:
            final_lanes=lane_lines
    filtered_detections = filter_detections_by_segmentation(detections, segmentation)
    min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)
    #for dis in min_distances:
        #assert(dis-0.1>0)
    #
    dist = np.abs(dist_to_plane(p_final, x, y, z))
    return dist, final_lanes, filtered_detections, min_distances
