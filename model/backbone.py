import keras
import keras.layers as layers
import keras.backend as K
from pointnet2 import (fully_connected, SampleAndGroup, pointnet_fp_module, 
                        pointnet_sa_module, crop)
import tensorflow as tf
from loss_helper import custom_gather

def votenet_backbone(pcd, feature_dims=0, n_centroid=[2048, 1024, 512, 256],random_sample=False):
    '''
    input: 
        pcd: B, N, 3+feature_dims
        n_centroid: number of centroids of each SA-Module, helps to create the ModelNet40 classifier
    output:
        xyz: B, 1024, 3
        feature: B, 1024, 256
        idx: B, 1024
    '''
    assert len(n_centroid)==4

    input = pcd
    if feature_dims>0:
        xyz0 = crop(2,0,3)(input)
        feature0 = crop(2,3,3+feature_dims)(input)
        use_feature = True
    else:
        xyz0 = input
        feature0 = input
        use_feature = False
    
    xyz_sa1, feature_sa1, idx_sa1 = pointnet_sa_module(xyz0, feature0, 
                                        n_centroid=n_centroid[0], radius=0.2, 
                                        n_samples=64, mlp=[64, 64, 128],
                                        use_xyz=True, use_feature=use_feature,
                                        random_sample=random_sample) # 2048
    xyz_sa2, feature_sa2, idx_sa2 = pointnet_sa_module(xyz_sa1, feature_sa1,
                                        n_centroid=n_centroid[1], radius=0.4,
                                        n_samples=32, mlp=[128, 128, 256],
                                        use_xyz=True, use_feature=True,
                                        random_sample=random_sample) # 1024
    xyz_sa3, feature_sa3, _ = pointnet_sa_module(xyz_sa2, feature_sa2, 
                                        n_centroid=n_centroid[2], radius=0.8, 
                                        n_samples=16, mlp=[128,128,256],
                                        use_xyz=True, use_feature=True,
                                        random_sample=random_sample) # 512
    xyz_sa4, feature_sa4, _ = pointnet_sa_module(xyz_sa3, feature_sa3, 
                                        n_centroid=n_centroid[3], radius=1.2, 
                                        n_samples=16, mlp=[128,128,256],
                                        use_xyz=True, use_feature=True,
                                        random_sample=random_sample) # 256
    
    feature_fp1 = pointnet_fp_module(xyz_sa3, xyz_sa4, feature_sa3, feature_sa4, [256,256])
    feature_fp2 = pointnet_fp_module(xyz_sa2, xyz_sa3, feature_sa2, feature_fp1, [256,256]) # B, 1024, 256
    
    idx = layers.Lambda(custom_gather, arguments={'indices':idx_sa2})(idx_sa1)
    return xyz_sa2, feature_fp2, idx

def make_classifier(num_points=2048, feature_dims=0, n_centroid=[256,128,64,32]):
    '''
    Create a classifier with votenet backbone. This is not aiming at a super accuray classifier
    Only for validing the backone.
    '''
    input = keras.Input((num_points, 3+feature_dims))
    xyz, features, idx = votenet_backbone(input, feature_dims=feature_dims, n_centroid=n_centroid)
    # make a quick and dirty classifier to validate votenet_backbone
    global_features = layers.MaxPool1D(pool_size=n_centroid[1])(features)
    net = layers.Flatten()(global_features) # B, 256
    net = layers.Dense(128, use_bias=False)(net)
    net = layers.BatchNormalization()(net)
    net = layers.ReLU()(net) 
    net = layers.Dropout(0.7)(net)

    net = layers.Dense(40, use_bias=True)(net) 
    net = layers.Softmax()(net)
    return keras.Model(inputs=input, outputs=net)

if __name__ ==  "__main__":
    input = keras.Input((20000,4))
    xyz, feature, idx = votenet_backbone(input, feature_dims=1)
    net = keras.Model(inputs=input, outputs=[xyz, feature])
    net.summary()
    keras.utils.plot_model(net, to_file='votenet_backbone.png', show_shapes=True, show_layer_names=True)