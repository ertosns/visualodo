from __future__ import print_function
#
import argparse
import os
import sys
import time
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from scipy import misc
import imageio
from segmentation.model import FCN8s, PSPNet50, ENet, ICNet
#
dir_path=os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(dir_path,'output/')
model_path = {'pspnet': os.path.join(dir_path,'model/pspnet50.npy'),
              'fcn': os.path.join(dir_path,'model/fcn.npy'),
              'enet': os.path.join(dir_path,'model/cityscapes/enet.ckpt'),
              'icnet': os.path.join(dir_path,'model/cityscapes/icnet.npy')}
#
model = ICNet()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Init tf Session
init = tf.global_variables_initializer()
sess.run(init)
model.load(model_path['icnet'], sess)
def segment(image):
    global model, config, sess, init
    model.set_input(image)
    print('model path: {}'.format(model_path['icnet']))
    #load model
    preds = model.forward(sess)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img=preds[0]
    #img_path=save_dir + 'icnet_'
    #imageio.imwrite(img_path, img)
    return img
#
