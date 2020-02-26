import keras
import keras.backend as K
from keras.layers import Layer
import keras.layers as layers
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
sys.path.append(os.path.join(ROOT_DIR, '../tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, '../tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, '../tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np

def random_sample(n_centroid, xyz):
    '''
    n_centroid: int32
    xyz: (batch, n_inputs, 3)
    Return:
        idx: (batch, n_centroid)
    '''
    distrib = tf.zeros(tf.shape(xyz)[:2])
    idx = tf.multinomial(distrib, n_centroid)
    idx = tf.cast(idx, tf.int32)
    return idx

class SampleAndGroup(Layer):
    def __init__(self, n_centroid, n_samples, radius,  random = False, use_xyz = True, use_feature=True, **kwargs):
        '''
        n_centroid: number of centroids
        n_sample: number of local samples around each centroid
        radius: radius of the ball when doing ball query
        random: True - use random sampling to choose centroids
                False - use farthest point sampling to choose centroids
        use_xyz: True - use coordinate of each points as features
                 False - not
        use_features: True - the second input tensor is high level feature of point
                      False - ignore the second input tensor 
        '''
        self.n_centroid = n_centroid
        self.n_samples = n_samples
        self.random = random
        self.use_xyz = use_xyz
        self.radius = radius
        self.use_feature = use_feature # input feature is none
        super(SampleAndGroup, self).__init__(**kwargs)
    def call(self, x):
        '''
        Input: List
            xyz : (batch, n_inputs, 3)
            features : (batch, n_inputs, channels)
        Output: List
            new_xyz: (batch_size, n_centroids, 3) TF tensor
            new_points: (batch_size, n_centroids, n_samples, 3+channel) TF tensor
            centroid_idx: (batch_size, n_centroids) TF tensor, indices of centroid
            grouped_xyz: (batch_size, n_centroids, n_samples, 3) TF tensor, normalized point XYZs
        '''
        xyz, features = x
        if self.random:
            centroid_idx = random_sample(self.n_centroid, xyz)
        else:
            centroid_idx = farthest_point_sample(self.n_centroid, xyz)
        new_xyz = gather_point(xyz, centroid_idx) # (batch, n_centroid, 3)
        idx, _ = query_ball_point(self.radius, self.n_samples, xyz, new_xyz)
        grouped_xyz = group_point(xyz, idx) # (batch_size, n_centroids, n_sample, 3)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,self.n_samples,1]) # translation normalization
        if self.use_feature: # can't use None type here
            grouped_points = group_point(features, idx) # (batch_size, n_centroid, n_samples, channels)
            if self.use_xyz:
                new_points = tf.concat([grouped_xyz, grouped_points], axis = -1) 
                # (batch_size, n_centroid, n_samples, channels + 3)
            else :
                new_points = grouped_points
        else :
            new_points = grouped_xyz
        return [new_xyz, new_points, centroid_idx, grouped_xyz]

    def compute_output_shape(self, input_shape):
        # input [(batch, n_inputs, 3), (batch, n_inputs, channels)]
        # output (batch, n_centroids, n_samples, 3)
        B, _, _ = input_shape[0]
        if self.use_feature:
            _, _, C = input_shape[1]
        else :
            C = 0
        return [(B, self.n_centroid, 3),(B, self.n_centroid, self.n_samples, 3+C),
            (B, self.n_centroid),(B, self.n_centroid, self.n_samples, 3)]
    

def mlp_layers(x, mlp, bn=True, relu6 = False):
    '''
    x: (batch, n_centroids, n_samples, channels) Tensor
    mlp: list of output channels
    '''
    for c in mlp:
        x = conv_bn_relu(x, c, bn, relu6)
    return x

def conv_bn_relu(x, output_channels, bn = True, relu6 = False):
    # according to MobileNet paper, ReLU6 is more robust when quantize the model
    x = layers.Conv2D(output_channels, [1,1], kernel_initializer='he_normal', use_bias=not bn)(x)
    if bn:
        x = layers.BatchNormalization()(x)
    if relu6:
        x = layers.ReLU(6.)(x)
    else :
        x = layers.ReLU()(x)
    return x

def pointnet_sa_module(xyz, features, n_centroid, radius, n_samples, mlp, bn=True, relu6=False, use_xyz = True, use_feature=True, random_sample=False):
    # sampling and ball grouping
    new_xyz, new_points, centroid_idx, _ = SampleAndGroup(n_centroid, n_samples, radius,
                                 random=random_sample, use_xyz=use_xyz, use_feature=use_feature)([xyz, features])
    # point feature embedding
    new_points = mlp_layers(new_points, mlp, bn, relu6) # (batch, n_centroid, n_samples, channels)
    # pooling in local regions
    new_points = layers.MaxPooling2D((1, n_samples))(new_points) # (batch, n_centroid, 1, channels)
    new_points = layers.Reshape((n_centroid, mlp[-1]))(new_points) # (batch, n_centroid, channels)
    return new_xyz, new_points, centroid_idx

def pointnet2_cls_ssg(num_class, num_points, num_dim = 3):
    '''
    input:  BxNx3
    output: Bxnum_class
    '''
    input = keras.Input((num_points,num_dim)) # (batch, num_points, num_dim)
    inp = input
    
    if num_dim > 3:
        l0_xyz = crop(2, 0, 3)(input)
        l0_points = crop(2, 3, num_dim)(input)
        use_feature = True
    else : 
        l0_xyz = input
        l0_points = input # useless
        # for the first stage, there is no high level feature, only coordinate
        use_feature = False
    
    l1_xyz, l1_points, _ = pointnet_sa_module(l0_xyz, l0_points,
                                    n_centroid=512, radius=0.2,
                                    n_samples=32, mlp=[64,64,128],
                                    bn=True, relu6=False, use_xyz=True,
                                    use_feature=use_feature, random_sample=False)
    
    l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points,
                                    n_centroid=128, radius=0.4,
                                    n_samples=64, mlp=[128,128,256],
                                    bn=True, relu6=False, use_xyz=True,
                                    use_feature=True, random_sample=False)
    '''
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
                                    n_centroid=32, radius=0.6,
                                    n_samples=32, mlp=[256,512,1024],
                                    bn=True, relu6=False, use_xyz=True,
                                    use_feature=True) 
    x = layers.GlobalMaxPooling1D()(l3_points)                                
    # at this stage, no sampling or grouping, use PointNet layer directly 
    # as Keras don't support None as input or output
    # the original implementation doesn't work here
    '''
    # try this instead
    x = l2_points
    x = layers.Reshape((-1,1,256))(x)
    x = mlp_layers(x, [256, 512, 1024])
    x = layers.GlobalMaxPooling2D()(x)

    # fullly connected layers
    # x = layers.Flatten()(x) # (Batch, :)
    x = fully_connected(x, 512, bn=True, relu6=False, activation=True)
    x = layers.Dropout(0.5)(x)
    x = fully_connected(x, 256, bn=True, relu6 = False, activation=True)
    x = layers.Dropout(0.5)(x)
    x = fully_connected(x, num_class, bn=False, activation=False) # no BN nor ReLU here
    x = layers.Softmax()(x)
    return keras.models.Model(inputs=inp, outputs=x)

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, bn=True, relu6=False):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            1: unknown
            2: known                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
    '''
    new_points1 = ThreeInterp()([xyz1, xyz2, points1, points2]) # B, n1, 1, c1+c2
    new_points1 = mlp_layers(new_points1, mlp, bn, relu6) # B, n1, 1, mlp[-1]
    new_points1 = layers.Reshape((-1, mlp[-1]))(new_points1)
    return new_points1

class ThreeInterp(Layer):
    def __init__(self, **kwargs):
        super(ThreeInterp, self).__init__(**kwargs)
    def call(self, x):
        '''
        Input:
            1: unknown
            2: known                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        '''
        xyz1, xyz2, points1, points2 = x
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum(1.0/dist, axis=2, keep_dims=True)
        norm = tf.tile(norm, [1,1,3])
        weight = (1.0/dist)/norm
        interpolated_points = three_interpolate(points2, idx, weight)
        # suppose points1 is not None
        new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B, n1, c1+c2
        new_points1 = tf.expand_dims(new_points1, 2) # B, n1, 1, c1+c2
        return new_points1
    def compute_output_shape(self, input_shape):
        B = input_shape[0][0]
        n1 = input_shape[0][1]
        c1 = input_shape[2][-1]
        c2 = input_shape[3][-1]
        return (B, n1, 1, c1+c2)

def fully_connected(x, output_dims, bn=True, relu6=False, activation=True):
    x = layers.Dense(output_dims)(x)
    if bn:
        x = layers.BatchNormalization()(x)
    if activation:
        if relu6:
            x = layers.ReLU(6.)(x)
        else: 
            x = layers.ReLU()(x)
    return x

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    # TODO: Rename it to Crop() so that it looks like a layer
    # TODO: make it able to crop x[:, start:] 
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return layers.Lambda(func)
