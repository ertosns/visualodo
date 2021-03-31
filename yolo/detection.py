from __future__ import division, print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import cv2

from yolo.utils.misc_utils import parse_anchors, read_class_names
from yolo.utils.nms_utils import gpu_nms
from yolo.utils.plot_utils import get_color_table
from yolo.utils.data_aug import letterbox_resize
import os
from yolo.model import yolov3

dir_path = os.path.dirname(os.path.realpath(__file__))
class_name_path=dir_path+"/data/coco.names"
anchor_path=dir_path+"/data/yolo_anchors.txt"
WEIGHTS_PATH=dir_path+"/data/darknet_weights/yolov3.ckpt"

# session initialization
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
num_class = len(classes)
color_table = get_color_table(num_class)
sess = tf.Session()
new_size=[416, 416]
input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
with tf.variable_scope('yolov3'):
    yolo_model = yolov3(num_class, anchors)
    pred_feature_maps = yolo_model.forward(input_data, False)
saver = tf.train.Saver()
saver.restore(sess, WEIGHTS_PATH)


def detections(img_ori, new_size=[416, 416]):
    """
    @param img_ori: opencv image
    @param new_size: required output size
    @return boxes_, scores_, labels_
    """


    #img_ori = cv2.imread(input_image)
    if letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, new_size[0], new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    #
    global  sess
    global yolo_model
    global input_data
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
    # rescale the coordinates to the original image
    if letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(new_size[1]))

    return boxes_, scores_, labels_
