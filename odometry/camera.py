import numpy as np
import cv2
import os

class Camera:
    def __init__(self, p):
        #assert(len(p)==12)
        #self.p = np.array(p).reshape(3,4)
        self.p=p
    def get_projection_matrices(self):
        return self.p

class FileCamera(Camera):
    def __init__(self, p, dataset_dir):
        #print('p: {}'.format(type(p)))
        #print('len: {}'.format(len(p)))
        super().__init__(p)
        self.dir=dataset_dir
        assert(os.path.exists(self.dir))
        path = os.walk(self.dir)
        for root, directories, files in path:
            count=0
            for f in files:
                if f.endswith('.png'):
                    count+=1
        self.count=int(count/2)
    def get_image(self,i):
        assert (i<self.size())
        left_file=os.path.join(self.dir, 'I1_'+format(i, '06d')+'.png')
        #print('left_file: {}'.format(left_file))
        assert(os.path.exists(left_file))
        right_file=os.path.join(self.dir, 'I2_'+format(i, '06d')+'.png')
        assert(os.path.exists(right_file))
        #
        left=cv2.imread(left_file)
        #print('left type: {}'.format(type(left)))
        #print('left shape: {}'.format(left.shape))
        right=cv2.imread(right_file)
        return left, right
    def size(self):
        return self.count
