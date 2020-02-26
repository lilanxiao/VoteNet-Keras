from keras import Input, layers
import keras.backend as K
from backbone import votenet_backbone
from voting_proposal import  voting_module, proposal_module
from keras.models import Model
from pointnet2 import crop
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, '../sunrgbd'))
from loss_helper import votenet_loss, custom_gather
from SUNRGBDDataset import MAX_NUM_OBJ
from dataset_model import SunrgbdDatasetConfig

def create_votenet_body(num_points, pcd_feature_dims, vote_factor,
                    num_class, num_head_bin, num_size_cluster, num_proposal, random_proposal=False):
    '''
    core of the VoteNet without layers for output decoding and loss calculation
    Arguments:
        pcd - Point cloud : B,num_points,3+feature_dims
    Returns:
        proposals_xyz:      B,num_proposals,3
        proposals_features: B,num_proposals,...
        seed_xyz:           B,num_seeds,3
        votes_xyz:          B,num_seeds*voting_factors,3
        seeds_inds:         B,num_seeds
    '''
    pcd = Input((num_points, 3+pcd_feature_dims),name='input_body')
    seeds_xyz, seeds_features, seeds_idx = votenet_backbone(pcd, pcd_feature_dims) # aggregate seed points
    votes_xyz, votes_features = voting_module(seeds_xyz, seeds_features, vote_factor) # voting
    # votes_features = layers.Lambda(K.l2_normalize,arguments={'axis':-1},name='vote_feature_normalize')(votes_features) # normalize vote features
    # seems unnecessary. explained in https://github.com/facebookresearch/votenet/issues/10
    proposals_xyz, proposals_features, _ = proposal_module(votes_xyz, votes_features, 
                                                        num_class, num_head_bin, 
                                                        num_size_cluster, num_proposal, 
                                                        random_sample=random_proposal) # prediction
    outputs = [proposals_xyz, proposals_features, seeds_xyz, votes_xyz, seeds_idx]
    return Model(inputs=pcd, outputs=outputs)

def decode_scores(args, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    '''
    function for decode output scores, should be wrapped as a Lambda layer
    ----------------
    Arguments:
        args[0]: proposal_xyz - B,num_proposals, 3
        args[1]: scores - B, num_proposals, 1, 2 + 3 + num_heading_bin*2 + num_size_cluster*4
    Returns as single list:
        objectness_score: B, num_proposals (use n below), 2
        center: B, n, 3
        heading_scores: B, n, num_heading_bin
        heading_residual_normalized: B, n, num_heading_bin
        heading_residual: B, n, num_heading_bin
        size_scores: B, n, num_size_cluster
        size_residual_normalized: B, n, num_size_cluster*3
        size_residual: B, n, num_size_cluster*3
        sem_cls_score: B, n, num_class
    '''
    proposals_xyz, net = args

    num_proposals = K.int_shape(net)[1]
    net = K.squeeze(net, 2) # squeeze to B, num_proposals, 2 + 3 + num_heading_bin*2 + num_size_cluster*4 + num_class

    objectness_score = net[:,:,0:2]

    center = proposals_xyz + net[:,:,2:5]
    heading_scores = net[:,:,5:5+num_heading_bin]
    heading_residuals_normalized = net[:,:, 5+num_heading_bin:5+num_heading_bin*2]
    heading_residuals = heading_residuals_normalized * (np.pi/num_heading_bin) # unused for loss

    size_score = net[:,:, 5+num_heading_bin*2:5+num_heading_bin*2 + num_size_cluster]
    size_residual_normalized = net[:,:, 5+num_heading_bin*2+num_size_cluster: 5+num_heading_bin*2+num_size_cluster*4]
    size_residual_normalized = K.reshape(size_residual_normalized, (-1, num_proposals, num_size_cluster, 3))
    mean_size_arr_extened = np.expand_dims(np.expand_dims(mean_size_arr, 0), 0)
    size_residual = K.tile(K.constant(mean_size_arr_extened, dtype="float32"), (1, num_proposals, 1, 1)) * size_residual_normalized
    # unused for loss

    sem_cls_scores = net[:,:,5+num_heading_bin*2+num_size_cluster*4:]
    return [objectness_score, center, heading_scores, heading_residuals_normalized, heading_residuals, 
            size_score, size_residual_normalized, size_residual, sem_cls_scores]

def create_votenet(num_points, pcd_feature_dims, vote_factor, 
                    num_class, num_head_bin, num_size_cluster, num_proposal, 
                    mean_size_arr, random_proposal, config):
    # inputs
    # point_cloud = Input((num_points, 3+pcd_feature_dims), name='point_cloud')
    center_label = Input((MAX_NUM_OBJ, 3), name='center_label')
    heading_class_label = Input((MAX_NUM_OBJ,), name='heading_class_label')
    heading_residual_label = Input((MAX_NUM_OBJ,), name='heading_residual_label')
    size_class_label = Input((MAX_NUM_OBJ,), name='size_class_label')
    size_residual_label = Input((MAX_NUM_OBJ,3), name='size_residual_label')
    sem_cls_label = Input((MAX_NUM_OBJ,), name='sem_cls_label')
    box_label_mask = Input((MAX_NUM_OBJ,), name='box_label_mask')
    vote_label = Input((num_points, 9), name='vote_label')
    vote_label_mask = Input((num_points,), name='vote_label_mask')

    # main body
    votenet_body = create_votenet_body(num_points, pcd_feature_dims, vote_factor, 
                                        num_class, num_head_bin,
                                        num_size_cluster, num_proposal, random_proposal)
    proposals_xyz, proposals_features, seeds_xyz, votes_xyz, seeds_idx = votenet_body.outputs
    decoded_scores = layers.Lambda(decode_scores, 
                                    arguments={'num_class':num_class, 
                                           'num_heading_bin':num_head_bin,
                                           'num_size_cluster':num_size_cluster,
                                           'mean_size_arr':mean_size_arr},
                                    name='decore_scores')([proposals_xyz, proposals_features])
    objectness_score, center, heading_scores, heading_residuals_normalized, \
    _, size_score, size_residual_normalized, _ , sem_cls_score = decoded_scores # unpack the decoded scores
    args_loss = [center_label, 
                heading_class_label,
                heading_residual_label,
                size_class_label,
                size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label,
                vote_label_mask,
                center,
                heading_scores, 
                heading_residuals_normalized,
                size_score,
                size_residual_normalized,
                sem_cls_score,
                seeds_xyz,
                seeds_idx,
                votes_xyz,
                objectness_score,
                proposals_xyz] # pack all arguments as a list
    # use Lambda layer to calculate the loss
    loss = layers.Lambda(votenet_loss, output_shape=(10,),
                        arguments={'config':config},
                        name='votenet_loss')(args_loss)
    return Model(inputs=[*votenet_body.inputs, 
                        center_label, 
                        heading_class_label,
                        heading_residual_label, 
                        size_class_label, 
                        size_residual_label, 
                        sem_cls_label,
                        box_label_mask,
                        vote_label,
                        vote_label_mask],
                outputs = loss,
                name='vote_net')

def predict_bbox(args, mean_size_arr):
    '''
    Using the decoded scores, predict one bbox for each proposal

    Arguments as a list:
        objectness_score: B, num_proposals (use n below), 2
        center: B, n, 3
        heading_scores: B, n, num_heading_bin
        heading_residual_normalized: B, n, num_heading_bin
        heading_residual: B, n, num_heading_bin
        size_scores: B, n, num_size_cluster
        size_residual_normalized: B, n, num_size_cluster*3
        size_residual: B, n, num_size_cluster*3
        sem_cls_score: B, n, num_class

    Outputs as a list:
        objectness_score_normalized: B,n,2
        center: B,n,3
        heading: B,n
        size: B,n,3
        sem_cls_score_normalized: B,n,num_class
    '''
    objectness_score = args[0]
    center = args[1]
    heading_scores = args[2]
    # heading_residuals_normalized = args[3] # unused
    heading_residuals = args[4]
    size_score = args[5]
    # size_residual_normalized = args[6] # unused
    size_residual = args[7]
    sem_cls_score = args[8]

    B = K.shape(objectness_score)[0]
    num_proposals = K.int_shape(heading_scores)[1]
    num_heading_bin = K.int_shape(heading_scores)[-1]
    num_size_cluster = K.int_shape(size_score)[-1]
    
    objectness_score_normalized = K.softmax(objectness_score, 2) # B,n,2

    # -----------heading------------------
    # in order to avoid the ugly indexing in Tensorflow, use one-hot, multiplication and sum instead  
    heading_scores_best_ind = K.argmax(heading_scores, axis=-1) # B,n
    heading_scores_musk = K.cast(K.one_hot(heading_scores_best_ind, num_heading_bin),'float32') # B,n,num_heading_bin
    heading_residuals_best = K.sum(heading_residuals * heading_scores_musk, axis=-1) # B,n
    heading = K.cast(heading_scores_best_ind,'float32') * K.cast(2*np.pi/num_heading_bin,'float32') + heading_residuals_best # B,n

    # ------------size---------------------
    size_score_best_ind = K.argmax(size_score, axis=-1) # B,n
    size_score_musk = K.cast(K.one_hot(size_score_best_ind, num_size_cluster),'float32') # B,n,num_size_cluster
    size_score_musk = K.tile(K.expand_dims(size_score_musk,-1), (1,1,1,3)) # B, n, num_size_cluster, 3
    size_residual = K.reshape(size_residual, (-1, num_proposals, num_size_cluster, 3)) # B, n, num_size_cluster, 3
    size_residual_best = K.sum(size_residual * size_score_musk, axis=2) # B,n,3

    size_anchors = K.constant(mean_size_arr, dtype='float32') # num_clusters,3
    size_anchors = K.expand_dims(K.expand_dims(size_anchors, 0),0) # 1,1,numc_clusters,3
    size_anchors = K.tile(size_anchors, (B, num_proposals, 1, 1)) # B,n,num_clusters,3
    size_anchors_best = K.sum(size_anchors*size_score_musk, axis=2) # B,n,3
    size = size_residual_best + size_anchors_best

    # -----------semantic classification--------
    sem_cls_score_normalized = K.softmax(sem_cls_score,-1) # B,n,num_class

    return [objectness_score_normalized, center, heading, size, sem_cls_score_normalized]

def create_votenet_inferencing(num_points, pcd_feature_dims, vote_factor, 
                    num_class, num_head_bin, num_size_cluster, num_proposal, 
                    mean_size_arr, random_proposal, config):
    '''
    Model used for inferencing

    Input Tensor:
        point cloud: B,num_points,num_features+3
    Output Tensor:
        objectness_score_normalized: B,num_proposals,2
        center: B,num_proposals,3
        heading: B,num_proposals
        size: B,num_proposals,3
        sem_class_score_normalized: B,num_proposals,num_class - semantic classification
        seeds_xyz: B,num_seeds,3
        votes_xyz: B,num_seeds*vote_factor,3
    '''

    # main body
    votenet_body = create_votenet_body(num_points, pcd_feature_dims, vote_factor, 
                                        num_class, num_head_bin,
                                        num_size_cluster, num_proposal, random_proposal) # inputs: point cloud
    proposals_xyz, proposals_features, seeds_xyz, votes_xyz, _ = votenet_body.outputs
    decoded_scores = layers.Lambda(decode_scores, 
                                    arguments={'num_class':num_class, 
                                           'num_heading_bin':num_head_bin,
                                           'num_size_cluster':num_size_cluster,
                                           'mean_size_arr':mean_size_arr},
                                    name='decore_scores')([proposals_xyz, proposals_features])
    bbox = layers.Lambda(predict_bbox, arguments={'mean_size_arr':mean_size_arr})(decoded_scores)
    votes_xyz = layers.Lambda(K.squeeze, arguments={'axis':2}, name='squeeze_votes_xyz')(votes_xyz)
    return Model(inputs=votenet_body.inputs, outputs=[*bbox, seeds_xyz, votes_xyz])

if __name__ == "__main__":
    from keras.utils import plot_model

    config = SunrgbdDatasetConfig()
    net = create_votenet(num_points=20000,
                        pcd_feature_dims=1,
                        vote_factor=1,
                        num_class=config.num_class,
                        num_head_bin=config.num_heading_bin,
                        num_size_cluster=config.num_size_cluster,
                        num_proposal=256,
                        mean_size_arr=config.mean_size_arr,
                        random_proposal=False,
                        config=config)
    net.summary()
    plot_model(net, to_file='votenet.png',show_shapes=True, show_layer_names=True)
