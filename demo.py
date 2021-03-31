from odometry.camera import FileCamera
from odometry.perception import *
from odometry.stereo import *
from matplotlib import pyplot as plt
from yolo.detection import detections
from segmentation.segment import *
from PIL import Image

# the demo is build using stero imaging from:
# camer calibration parameters for mrpt/karlsruhe_dataset is taken from https://www.mrpt.org/Karlsruhe_Dataset_Rawlog_Format
# DIR is set to Karlsruhe dataset
DIR='/opt/stereo/2010_03_09_drive_0019/2010_03_09_drive_0019_Images'
#left camera
P1=P1.reshape(3,4)
#P1=np.array([6.452401e+02, 0.000000e+00, 6.359587e+02, 0.000000e+00, 0.000000e+00, 6.452401e+02, 1.941291e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00])
left_cam = FileCamera(P1, DIR)
left_P = left_cam.get_projection_matrices()
k_left, _, t_left = decompose_projection_matrix(left_P)
#k_left= np.array([8.941981e+02, 0.000000e+00, 6.601406e+02, 0.000000e+00, 8.927151e+02, 2.611004e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape((3,3))
#
#right camera
P2=P2.reshape(3,4)
#P2=np.array([6.452401e+02, 0.000000e+00, 6.359587e+02, -3.682632e+02, 0.000000e+00, 6.452401e+02, 1.941291e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00])
right_cam = FileCamera(P2, DIR)
right_P = right_cam.get_projection_matrices()
k_right, _, t_right = decompose_projection_matrix(right_P)
#k_right=np.array([8.800704e+02, 0.000000e+00, 6.635881e+02, 0.000000e+00, 8.798504e+02, 2.690108e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape((3,3))

L=left_cam.size()
for i in range(left_cam.size()):
    left, right=left_cam.get_image(i)
    #print('left shape: {}, right shape: {}'.format(left.shape, right.shape))
    disp_left = compute_left_disparity_map(left, right)
    #print('disp shape: {}'.format(disp_left.shape))
    depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)
    #TODO camera calibration parameter seems wrong!
    depth_map_left=np.abs(depth_map_left*1000*1000*1000 * 1000 * 1000 * 10)
    #print('disp_map_left shape: {}'.format(depth_map_left.shape))
    plt.imsave('./out/depth/depth_'+str(i)+'.png', depth_map_left)
    plt.imsave('./out/disp/disp_'+str(i)+'.png', disp_left)
    #print("depth sum: {}".format(np.sum(depth_map_left)))
    assert(np.sum(depth_map_left)>0.1)
    #plt.imshow(left)
    #plt.show()
    boxes, scores, labels = detections(left)
    #print("boxes: {}, scores: {}, labesl: {}".format(boxes, scores, labels))
    num_dets = len(scores)
    dets = []
    for  item in range(num_dets):
        #print('labels: {}, boxes: {}, scores: {}'.format(labels[i], boxes[i], scores[i]))
        dets.append(np.concatenate((np.array([labels[item]]), np.array(boxes[item]), np.array([scores[item]]))))
        #[np.squeeze(np.dstack(np.stack(labels[i], boxes[i], scores[i])))
    segmentation = segment(left)
    #segmentation=segmentation.astype('int32')
    #print('type: {}, value: {}'.format(type(segmentation), segmentation))
    segmentation_img = Image.fromarray(segmentation, 'RGB')
    shape=left.shape
    segmentation_img=segmentation_img.resize((shape[1], shape[0]))
    segmentation_img = np.array(segmentation_img)
    #print('segmentation shape: {}'.format(segmentation.shape))
    segmentation_grey=segmentation[:,:,0]
    #segmentation_grey.reshape(left.shape[0:2])
    shape=left.shape
    segmentation_grey=cv2.resize(segmentation_grey, dsize=(shape[1], shape[0]))
    segmentation_grey=np.array(segmentation_grey)
    #print('segmentation: {}'.format(segmentation_grey))
    #print('grey segmentation: {}'.format(segmentation_grey.shape))
    dist, final_lanes, filtered_detection, min_distance = perceive(left, segmentation_grey, depth_map_left, k_left, dets)
    #print('filtered_detection: {}, final_lanes: {}, dets: {}'.format(filtered_detection, final_lanes, dets))
    #
    ground_mask = np.zeros(dist.shape)
    ground_mask[dist < 0.1] = 1
    ground_mask[dist > 0.1] = 0
    ZZ=np.zeros(ground_mask.shape)
    ground_mask = np.stack([ground_mask, ZZ, ZZ], axis=-1)
    shape=left.shape
    out = np.zeros(shape)
    for w in  range(shape[0]):
        for h in range(shape[1]):
            for d in range(shape[2]):
                if ground_mask[w,h,d]==1:
                    out[w,h,d] = segmentation_img[w,h,d]
                else:
                    out[w,h,d] = left[w,h,d]
    out=out.astype('int32')
    img_out=vis_lanes(out, final_lanes)
    img_out=vis_object_detection(img_out, filtered_detection, min_distance)
    img_final_path='./out/final/fin'+str(i)+'.png'
    print('saving image: {}'.format(img_final_path))
    plt.imsave(img_final_path, img_out)
