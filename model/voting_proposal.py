import keras
import keras.backend as K
import keras.layers as layers
from pointnet2 import mlp_layers, conv_bn_relu, crop, SampleAndGroup, pointnet_sa_module
from loss_helper import check_shape

def voting_module(seed_xyz, seed_features, vote_factor):
    '''
    Inputs:
        seed_xyz: (B, num_seed, 3) or (B, num_seed, 1, 3)
        seed_features: (B, num_seed, seed_feature_dim) or (B, num_seed, 1, seed_feature_dim)
    Returns:
        vote_xyz: # (B, num_seed * vote_factor, 1,  3) 
        vote_features: (B, num_seed * vote_factor, 1,  seed_feature_dim)
    '''
    seed_feature_dim = K.int_shape(seed_features)[-1]
    if len(K.int_shape(seed_xyz)) == 3:
        seed_xyz = layers.Reshape((-1,1,3))(seed_xyz) #(B, num_seed, 1, 3)
        seed_features = layers.Reshape((-1,1,seed_feature_dim))(seed_features) #(B, num_seed, 1, seed_feature_dim)
    num_seed = K.int_shape(seed_xyz)[-3]
    num_vote = num_seed * vote_factor
    net = conv_bn_relu(seed_features, seed_feature_dim) # B, num_seed, 1, 3
    net = conv_bn_relu(net, seed_feature_dim) # B, num_seed, 1, 3
    net = layers.Conv2D(filters=(seed_feature_dim + 3)*vote_factor, kernel_size=[1,1])(net)
    # (B, num_seed, 1, (3+seed_feature_dim)*vote_factor)
    if vote_factor != 1:
            net = layers.Reshape((-1,vote_factor,3+seed_feature_dim))(net) # (B, num_seed, vote_factor, (3+seed_feature_dim))
            seed_xyz = layers.Lambda(K.tile, arguments={'n':(1, 1, vote_factor, 1)})(seed_xyz) # (B, num_seed, vote_factor, 3)
            seed_features = layers.Lambda(K.tile, arguments={'n':(1, 1, vote_factor, 1)})(seed_features) # (B,num_seed,vote_factor, dims)
    # voting for the centroid of object
    offset = crop(3, 0, 3)(net) # (B, num_seed, vote_factor, 3)
    # vote_xyz = seed_xyz + offset # (B, num_seed, vote_factor, 3)
    vote_xyz = layers.Add()([seed_xyz, offset])
    vote_xyz = layers.Reshape((num_vote, 1, 3))(vote_xyz) # (B, num_seed * vote_factor, 1,  3) 
    # feature residual
    residual_features = crop(3, 3, seed_feature_dim+3)(net) # (B, num_seed, vote_factor, seed_feature_dim)
    # vote_features = seed_features + residual_features # # (B, num_seed, vote_factor, seed_feature_dim)
    vote_features = layers.Add()([seed_features, residual_features])
    vote_features = layers.Reshape((num_vote, 1, seed_feature_dim))(vote_features) # (B, num_seed * vote_factor, 1,  seed_feature_dim)
    return vote_xyz, vote_features

def voting_module_simple(seed_xyz, seed_features):
    '''
    suppose vote_factor is 1
    Input:
        seed_xyz: (B, num_seed, 3) or (B, num_seed, 1, 3)
        seed_features: (B, num_seed, seed_feature_dim) or (B, num_seed, 1, seed_feature_dim)
    Returns:
        vote_xyz: (B, num_seed, 1,  3) 
        vote_features: (B, num_seed, 1,  seed_feature_dim)
    '''
    seed_feature_dim = K.int_shape(seed_features)[-1]
    if len(K.int_shape(seed_xyz)) == 3:
        seed_xyz = layers.Reshape((-1,1,3))(seed_xyz) #(B, num_seed, 1, 3)
        seed_features = layers.Reshape((-1,1,seed_feature_dim))(seed_features) #(B, num_seed, 1, seed_feature_dim)

    net = conv_bn_relu(seed_features, seed_feature_dim) # B, num_seed, 1, 3
    net = conv_bn_relu(net, seed_feature_dim) # B, num_seed, 1, 3
    net = layers.Conv2D(filters=(seed_feature_dim + 3), kernel_size=[1,1])(net) #(B, num_seed, 1, (3+seed_feature_dim))

    # voting for the centroid of object
    offset = crop(3, 0, 3)(net) # (B, num_seed, 1, 3)
    vote_xyz = layers.Add()([seed_xyz, offset]) # (B, num_seed, 1, 3)
    # feature residual
    residual_features = crop(3, 3, seed_feature_dim+3)(net) # (B, num_seed, vote_factor, seed_feature_dim)
    vote_features = layers.Add()([seed_features, residual_features]) # # (B, num_seed, vote_factor, seed_feature_dim)
    vote_features = layers.Reshape((-1, 1, seed_feature_dim))(vote_features)
    return vote_xyz, vote_features

def proposal_module(vote_xyz, vote_features, num_class, num_head_bin, 
                    num_size_cluster, num_proposal, random_sample=False):
    '''
    Return:
        xyz: proposal_xyz
        scores: scores
        idx: proposal index
    '''
    seed_feature_dim = K.int_shape(vote_features)[-1]
    vote_xyz = layers.Reshape((-1, 3))(vote_xyz)
    vote_features = layers.Reshape((-1, seed_feature_dim))(vote_features) # squeeze the tensors
    xyz, features, idx = pointnet_sa_module(vote_xyz,vote_features, mlp=[128,128,128], 
                                            n_centroid=num_proposal, n_samples=16, 
                                            radius=0.3, use_feature=True, use_xyz=True, 
                                            random_sample=random_sample)
    features = layers.Reshape((-1,1,128))(features) # 128 = mlp[-1]
    net = conv_bn_relu(features, 128)
    net = conv_bn_relu(net, 128)
    # objectness scores: 2
    # center residual: 3
    # heading: classification + residual = 2*num_head_bin
    # size: classification + residual = 4*num_size_cluster
    # object classification: num_class
    output_dims = 2 + 3 + num_head_bin*2 + num_size_cluster*4 + num_class
    scores = layers.Conv2D(filters=output_dims, kernel_size=(1,1))(net)
    return xyz, scores, idx