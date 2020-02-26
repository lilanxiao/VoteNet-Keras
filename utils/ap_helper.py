""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../sunrgbd'))
sys.path.append(os.path.join(ROOT_DIR, '../model'))
from eval_det import eval_det_cls, eval_det_multiprocessing
from eval_det import get_iou_obb
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from box_util import get_3d_box
from sunrgbd_utils import extract_pc_in_box3d

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def parse_predictions(objectness_score_normalized, center, heading, size, sem_class_scores, 
                    conf_thresh, nms_iou, num_class, per_class_proposal=False, cls_nms=False):
    '''
    Parse predictions to OBB parameters and suppress overlapping boxes
    NOTE: inputs are numpy array, not Tensorflow tensor

    Args:
        objectness_score_normalized: B,num_proposals,2
        center: B,num_proposals,3
        heading: B,num_proposals
        size: B,num_proposals,3
        sem_class_scores: B,num_proposals,num_class
        conf_thresh: threshhold of objectness
        nms_iou: threshold of IoU
    
    Returns:
        pred_mask: B,K - 0/1
        batch_pred_map_cls: a list (len: batch_size) of list (len: num of predictions per sample) of tuples of 
                            pred_cls, pred_box and conf (0-1)
    '''
    obj_prob = objectness_score_normalized[:,:,1] # B,K. score for positive
    sem_class = np.argmax(sem_class_scores,axis=-1) # B,K
    B, K = sem_class.shape # batch size, num_proposals
    center_upright_camera = flip_axis_to_camera(center)
    corners_3d_upright_camera = np.zeros((B,K,8,3))
    for i in range(B):
        for j in range(K):
            if heading[i,j] > np.pi:
                heading[i,j] -= 2*np.pi
            if np.all(size[i,j] == np.array([0,0,0])):
                print("size zero!") # for debugging
            corners_3d_upright_camera[i,j] = get_3d_box(size[i,j], heading[i,j], center_upright_camera[i,j,:])

    if cls_nms:
        pred_mask = np.zeros((B,K))
        for i in range(B):
            boxes_3d_with_prob = np.zeros((K,8)) # bbox for one scene
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(corners_3d_upright_camera[i,j,:,0]) # x_min
                boxes_3d_with_prob[j,1] = np.min(corners_3d_upright_camera[i,j,:,1]) # y_min
                boxes_3d_with_prob[j,2] = np.min(corners_3d_upright_camera[i,j,:,2]) # z_min
                boxes_3d_with_prob[j,3] = np.max(corners_3d_upright_camera[i,j,:,0]) # x_max
                boxes_3d_with_prob[j,4] = np.max(corners_3d_upright_camera[i,j,:,1]) # y_max
                boxes_3d_with_prob[j,5] = np.max(corners_3d_upright_camera[i,j,:,2]) # z_max
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
                boxes_3d_with_prob[j,7] = sem_class[i,j]
            # use aixs aligned bbox to do the NMS
            pick = nms_3d_faster_samecls(boxes_3d_with_prob, nms_iou) # get index of picked bbox
            assert len(pick)>0
            pred_mask[i, pick]=1
    else:
        pred_mask = np.zeros((B,K))
        for i in range(B):
            boxes_3d_with_prob = np.zeros((K,7))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
            pick = nms_3d_faster(boxes_3d_with_prob, nms_iou) # get index of picked bbox
            assert len(pick)>0
            pred_mask[i, pick]=1

    batch_pred_map_cls = []
    for i in range(B):
        if per_class_proposal:
            cur_list = []
            for ii in range(num_class):
                cur_list += [(ii, corners_3d_upright_camera[i,j], sem_class_scores[i,j,ii]*obj_prob[i,j]) \
                        for j in range(K) if pred_mask[i,j]==1 and obj_prob[i,j]>conf_thresh]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(sem_class[i,j].item(), corners_3d_upright_camera[i,j], obj_prob[i,j]) \
                for j in range(K) if pred_mask[i,j]==1 and obj_prob[i,j]>conf_thresh])
    return batch_pred_map_cls

def parse_groundtruths(center_label, heading_class_label, heading_residual_label, size_class_label,
                        size_residual_label, sem_cls_label, box_label_mask, config, debug = False):
    '''
    Args:
        center_label:           B, MAX_NUM_OBJ, 3
        heading_class_label:    B, MAX_NUM_OBJ,
        heading_residual_label: B, MAX_NUM_OBJ, 
        size_class_label:       B, MAX_NUM_OBJ,
        size_residual_label:    B, MAX_NUM_OBJ, 3
        sem_cls_label:          B, MAX_NUM_OBJ, 
        box_label_mask:         B, MAX_NUM_OBJ,
    '''
    B,K2,_ = center_label.shape
    gt_corners_3d_upright_camera = np.zeros((B,K2,8,3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:,:,0:3])
    for i in range(B):
        for j in range(K2):
            if box_label_mask[i,j] == 0: continue
            heading_angle = config.class2angle(heading_class_label[i,j], heading_residual_label[i,j])
            box_size = config.class2size(int(size_class_label[i,j]), size_residual_label[i,j])
            gt_corners_3d_upright_camera[i,j] = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i,j,:])
    
    batch_gt_map_cls = []
    for i in range(B):
        if debug: # create fake prediction for debugging
            batch_gt_map_cls.append([(sem_cls_label[i,j].item(), gt_corners_3d_upright_camera[i,j]*np.random.rand(), np.random.rand()) for j in range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i,j]==1])
        else:
            batch_gt_map_cls.append([(sem_cls_label[i,j].item(), gt_corners_3d_upright_camera[i,j]) for j in range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i,j]==1])
    return batch_gt_map_cls

class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision'%(clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall'%(clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
 