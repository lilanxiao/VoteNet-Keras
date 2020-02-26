# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

Modified by Lanxiao Li, 2020
"""
import os
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_utils
import sunrgbd_utils
from dataset_model import SunrgbdDatasetConfig
from keras.utils import Sequence

#--------------------user configuration----------------------------------
DC = SunrgbdDatasetConfig()
MAX_NUM_OBJ = 64    # max number of objects labels per scene
MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5]) # mean color value of sunrgbd
DATA_ROOT = "../data" #
#------------------------------------------------------------------------

class SunrgbdDetectionVotesDataset(Sequence):
    def __init__(self, split_set='train', batch_size=8, num_points=20000, shuffle=False,
        use_color=False, use_height=True, use_v1=False,
        augment=False, use_features=False, scan_idx_list=None):
        if use_v1:
            self.data_path = os.path.join(DATA_ROOT,
                'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_%s'%(split_set))
        else:
            self.data_path = os.path.join(DATA_ROOT,
                'sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_%s'%(split_set))
        self.raw_data_path = os.path.join(DATA_ROOT, 'sunrgbd/sunrgbd_trainval')
        self.scan_names= sorted(list(set([os.path.basename(x)[0:6]\
            for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            # if given, only use listed scans
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.batch_size = batch_size
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.split = split_set
        self.use_features = use_features # to do: use low level features
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.scan_names))
        if self.shuffle:
            np.random.shuffle(self.indices)
        np.random.seed()
    
    def __data_generation_(self, idx):
        '''
        Returns:
            point_cloud:            N,3+C 
            center_label:           MAX_NUM_OBJ, 3
            heading_class_label:    MAX_NUM_OBJ,
            heading_residual_label: MAX_NUM_OBJ, 
            size_class_label:       MAX_NUM_OBJ,
            size_residual_label:    MAX_NUM_OBJ, 3
            sem_cls_label:          MAX_NUM_OBJ, 
            box_label_mask:         MAX_NUM_OBJ,
            vote_label:             N, 9
            vote_label_mask:        N,
        '''
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc'] # N,6
        # Bounding boxes (K,8)
        # [0:3]: centroid coordinate. x,y,z
        # [3:6]: size. height, width, height
        # [6]: heading angle
        # [7]: class one hot label        
        bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy') # K,8: 
        # Votes (N, 10) --3 votes and 1 vote mask
        # [0]: this point is in a bounding box or not (0/1)
        # [1:4],[4:7],[7:10]: if point is not in any bounding box, all zeros; 
        # else the offset to bouding box center
        # one point can be assigned to at maximal 3 bounding boxes
        point_votes = np.load(os.path.join(self.data_path, scan_name)+'_votes.npz')['point_votes'] # Nx10

        if not self.use_color:
            point_cloud = point_cloud[:,0:3] # x,y,z
        else:
            point_cloud = point_cloud[:, 0:6] # x,y,z,r,g,b
            point_cloud[:,3] = point_cloud[:,3:] - MEAN_COLOR_RGB

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2], 0.99) # 0.99% of all height. wired number...
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height,1)], 1) # N,4 or N,7
        
        #-------------data augmentation-------------
        if self.augment:
            if np.random.rand() > 0.5:
                # flipping along YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                bboxes[:,0] = bboxes[:,0] * -1 
                bboxes[:,6] = np.pi - bboxes[:,6]
                point_votes[:,[1,4,7]] = -1 * point_votes[:,[1,4,7]]
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.rand()*np.pi/3) - np.pi/6 # -30~30 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle)
            point_votes_end = np.zeros_like(point_votes)
            # first, rotate votes "with" the point_cloud
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))
            point_votes_end[:,4:7] = np.dot(point_cloud[:,0:3] + point_votes[:,4:7], np.transpose(rot_mat))
            point_votes_end[:,7:10] = np.dot(point_cloud[:,0:3] + point_votes[:,7:10], np.transpose(rot_mat))
            # then, rotate the point cloud alone
            point_cloud[:, 0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            bboxes[:,0:3] = np.dot(bboxes[:,0:3], np.transpose(rot_mat))
            bboxes[:,6] -= rot_angle    # the original angle is NOT filpped
            # finally, restore the point_votes by recalculate the offset
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]
            point_votes[:,4:7] = point_votes_end[:,4:7] - point_cloud[:,0:3]
            point_votes[:,7:10] = point_votes_end[:,7:10] - point_cloud[:,0:3]

            # augment the color
            if self.use_color:
                rgb_color = point_cloud[:,3:6] + MEAN_COLOR_RGB # restore color to 0~1
                rgb_color *= (1+0.4*np.random.rand(3)-0.2) # random scale brightness 80% ~ 120%
                rgb_color += (0.1*np.random.rand(3)-0.05) # random shift
                rgb_color += np.expand_dims(np.random.rand(point_cloud.shape[0])*0.05-0.025,-1) #random jitter
                rgb_color = rgb_color - MEAN_COLOR_RGB
                rgb_color *= np.expand_dims(np.random.rand(point_cloud.shape[0])>0.3, -1) # drop 30% colors
            
            # scale the size
            scale_ratio = np.random.rand()*0.3 + 0.85 # 0.85 ~ 1.15
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:,0:6] *= scale_ratio
            point_votes[:,1:-1] *= scale_ratio
            if self.use_height:
                point_cloud[:,-1] *= scale_ratio

            # shift the point cloud -0.5~0.5
            offset = np.random.rand(3) - 0.5
            offset = np.expand_dims(offset, 0)
            point_cloud[:,0:3] += offset
            bboxes[:,0:3] += offset
            # shifting doesn't change: size, votes, height

        # ------------labels------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ,3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0],:] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = bbox[3:6]*2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size
        
        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((MAX_NUM_OBJ,6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:,0])
            ymin = np.min(corners_3d[:,1])
            zmin = np.min(corners_3d[:,2])
            xmax = np.max(corners_3d[:,0])
            ymax = np.max(corners_3d[:,1])
            zmax = np.max(corners_3d[:,2])
            # 0:3 - centers
            # 3:6 - size
            target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i,:] = target_bbox

        point_cloud, choice = pc_utils.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choice,0]
        point_votes = point_votes[choice,1:]

        center_label = target_bboxes.astype(np.float32)[:,:3]
        heading_class_label = angle_classes.astype(np.int64)
        heading_residual_label = angle_residuals.astype(np.float32)
        size_class_label = size_classes.astype(np.int64)
        size_residual_label = size_residuals.astype(np.float32)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9

        sem_cls_label = target_bboxes_semcls.astype(np.int64)
        box_label_mask = target_bboxes_mask.astype(np.float32)
        vote_label = point_votes.astype(np.float32)
        vote_label_mask = point_votes_mask.astype(np.int64)

        return [point_cloud.astype(np.float32), \
            center_label, \
            heading_class_label, \
            heading_residual_label, \
            size_class_label, \
            size_residual_label, \
            sem_cls_label, \
            box_label_mask, \
            vote_label, \
            vote_label_mask]
    
    def __len__(self):
        return int(np.floor(len(self.indices)/self.batch_size))
    
    def __getitem__(self, index):
        # NOTE:
        # numpy get duplicated random numbers in different bacth when using multiprocessing. Not sure how to deal with it.
        # https://github.com/pytorch/pytorch/issues/5059
        # if you uncomment the following line here and don't reset random seed, you will see duplicated numbers
        # print(np.random.rand())
        # np.random.seed() # reset the random seed explicitly might help
        B = self.batch_size
        C = 1*int(self.use_height) + 3*int(self.use_color) # number of featurs
        batch_idx = self.indices[index*B: (index+1)*B] 
        y_batch = np.zeros((B,10))
        # allocate memory for a batch
        point_cloud_b = np.zeros((B, self.num_points, C+3), dtype=np.float32)
        center_label_b = np.zeros((B, MAX_NUM_OBJ, 3), dtype=np.float32)
        heading_class_label_b = np.zeros((B, MAX_NUM_OBJ), dtype=np.int64)
        heading_residual_label_b = np.zeros((B, MAX_NUM_OBJ), dtype=np.float32)
        size_class_label_b = np.zeros((B, MAX_NUM_OBJ), dtype=np.int64)
        size_residual_label_b = np.zeros((B, MAX_NUM_OBJ, 3), dtype=np.float32)
        sem_cls_label_b = np.zeros((B, MAX_NUM_OBJ), dtype=np.int64)
        box_label_mask_b = np.zeros((B, MAX_NUM_OBJ), dtype=np.float32)
        vote_label_b = np.zeros((B, self.num_points, 9), dtype=np.float32)
        vote_label_mask_b = np.zeros((B,self.num_points), dtype=np.float32)

        for i in range(B):
            idx = batch_idx[i]
            data = self.__data_generation_(idx)
            point_cloud_b[i] = data[0]
            center_label_b[i] = data[1]
            heading_class_label_b[i] = data[2]
            heading_residual_label_b[i] = data[3]
            size_class_label_b[i] = data[4]
            size_residual_label_b[i] = data[5]
            sem_cls_label_b[i] = data[6]
            box_label_mask_b[i] = data[7]
            vote_label_b[i] = data[8]
            vote_label_mask_b[i] = data[9]

        return [point_cloud_b, center_label_b, \
                heading_class_label_b, heading_residual_label_b, \
                size_class_label_b, size_residual_label_b, \
                sem_cls_label_b, box_label_mask_b, \
                vote_label_b, vote_label_mask_b], \
                y_batch

if __name__ == "__main__":
    dataset = SunrgbdDetectionVotesDataset(batch_size=8)
    counter = 0
    for i in range(len(dataset)):
        x,_ = dataset[i]
        box_label_mask = x[7]
        for j in range(8):
            if np.sum(box_label_mask[j])==0:
                counter += 1
    print(counter)
