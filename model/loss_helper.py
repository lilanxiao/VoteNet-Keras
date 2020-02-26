import keras
import keras.layers as layers
import keras.backend as K
import numpy as np
import tensorflow as tf

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def huber_loss(error, delta=1.0):
    abs_error = K.abs(error)
    quadratic = K.clip(abs_error, max_value=delta, min_value=-1)
    linear = (abs_error -quadratic)
    loss = 0.5*K.square(quadratic) + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = K.int_shape(pc1)[1]
    M = K.int_shape(pc2)[1]
    pc1_expand = K.tile(K.expand_dims(pc1, 2),[1,1,M,1])
    pc2_expand = K.tile(K.expand_dims(pc2, 1),[1,N,1,1])
    pc_diff = pc1_expand - pc2_expand
    if l1smooth:
        pc_dist = K.sum(huber_loss(pc_diff, delta),axis=-1)
    elif l1:
        pc_dist = K.sum(K.abs(pc_diff), axis=-1)
    else:
        pc_dist = K.sum(K.square(pc_diff), axis=-1) # (B,N,M)
    dist1 = K.min(pc_dist, axis=2, keepdims=False) #(B,N)
    idx1 = K.argmin(pc_dist, axis=2) # 
    dist2 = K.min(pc_dist, axis=1, keepdims=False) #(B,M)
    idx2 = K.argmin(pc_dist, axis=1)
    return dist1, idx1, dist2, idx2

def compute_vote_loss(seed_xyz, vote_xyz, seed_inds, vote_label, vote_label_mask):
    '''
    Inputs:
        seed_xyz: B, num_seed, 3
        vote_xyz: B, num_seed*vote_factor, 3
        seed_ins: B, num_seed in [0, num_points-1]
        vote_label: B, num_total_points, 3*3
        vote_label_mask: B, N
    ----------------------------
    Note: 
        this function is complicated because it has to consider the case that one seed can predict
        several votes. Would be significantly simpler if each seed predicts one vote
        That is, only take the nearst object centroid to calculate the Ground Truth vote for each point. 
    '''
    batch_size = K.shape(seed_xyz)[0]
    num_seed = K.int_shape(seed_xyz)[1]
    num_vote = K.int_shape(vote_xyz)[1]
    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = custom_gather(vote_label_mask, seed_inds)      # vote mask for seed points
    seed_inds_expand = K.tile(K.reshape(seed_inds, (-1,num_seed,1)),[1,1,3*GT_VOTE_FACTOR])
    seed_gt_votes = custom_gather_v2(vote_label, seed_inds_expand) # vote label for seed points
    seed_gt_votes += K.tile(seed_xyz,[1,1,3]) # B,num_seed, 9

    # compute the min of min distance
    vote_xyz_reshape = K.reshape(vote_xyz, (batch_size*num_seed, int(num_vote/num_seed), 3)) #B*num_seed, vote_factor,3
    seed_gt_votes_reshape = K.reshape(seed_gt_votes, (batch_size*num_seed, GT_VOTE_FACTOR, 3))
    # a predicted vote to nowhere us not penalised as long as there is a good vote near the GT vote
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    # Basically, it pairs votes of each point with the groundtruth.
    # if vote factor is one, this does nothing.
    votes_dist = K.min(dist2, axis=1) # (B*num_seed,vote_factor) to (B*num_seed,)

    # only the best vote of the seed points contributes to the total loss
    votes_dist = K.reshape(votes_dist, (batch_size, num_seed))
    vote_loss = K.sum(votes_dist * K.cast(seed_gt_votes_mask, dtype='float32'))/(K.cast(K.sum(seed_gt_votes_mask),dtype='float32')+K.epsilon())
    return vote_loss

def compute_objectness_loss(proposal_xyz, center_label, objectness_score):
    '''
    Args: 
        aggregated_vote_xyz: (B, K1, 3) proposal
        center_label: (B, K2, 3+1+3) GT
        objectness_score: (B, K1, 2)

    Return:
        loss: scalar
        objectness_label: (B, K1) - predicted center is near of a groud truth of not
        objectness_mask: (B, K1) - Care of not care
        obeject_assignment: (B, K1)- Index of corresponding GT-center for each prediction 
    '''
    gt_center = center_label[:,:,:3]
    dist1, ind1, dist2, _ = nn_distance(proposal_xyz, gt_center) 
    # distance between nearest predicted center and GT

    sqrt_dist1 = K.sqrt(dist1 + K.epsilon())
    objectness_label = K.greater(K.ones_like(sqrt_dist1)*NEAR_THRESHOLD, sqrt_dist1)
    objectness_label = K.cast(objectness_label, 'int32')
    # label = 1, if sqrt_dist < NEAR_THRESHOLD
    objectness_mask_1 = K.cast(K.greater(K.ones_like(sqrt_dist1)*NEAR_THRESHOLD, sqrt_dist1), 'float32')
    objectness_mask_2 = K.cast(K.greater(sqrt_dist1, K.ones_like(sqrt_dist1)*FAR_THRESHOLD), 'float32')
    objectness_mask = objectness_mask_1 + objectness_mask_2
    # mask = 1, if sqrt_dist < NEAR_THRESHHOLD or sqrt_dist > FAR_THRESHOLD
    loss = weighted_crossentropy(K.one_hot(objectness_label, 2), objectness_score, OBJECTNESS_CLS_WEIGHTS)
    loss = K.sum(loss * objectness_mask) / (K.sum(objectness_mask)+K.epsilon())
    object_assignment = ind1
    return loss, objectness_label, objectness_mask, object_assignment


def weighted_crossentropy(y_true, y_pred, weights):
    '''
    apply softmax and weighted negative logit loss
    ------------------------
    Args: 
        y_true: (B, num_proposal, num_class) - one hot label
        y_pred: (B, num_proposal, num_class) - prediction from conv layers
        weight: (num_class, )
    
    Return:
        loss: (B, P)
    '''
    weights = K.constant(weights) # C
    y_true = K.cast(y_true,'float32')
    B = K.shape(y_true)[0]
    P = K.shape(y_true)[1]
    y_pred_softmax = K.softmax(y_pred, -1) # apply softmax
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1 - K.epsilon()) # B, P, C
    weights = K.expand_dims(K.expand_dims(weights, 0), 0) # expand weights to 1,1,C
    weights = K.tile(weights, [B,P,1]) # tile weights to B,P,num_class    
    loss = y_true * K.log(y_pred_softmax) * weights # B, P, C
    loss = -K.sum(loss, -1) # B, P
    return loss

def compute_box_and_sem_cls_loss(object_assignment, pred_center, center_label, box_label_mask, objectness_label, 
                                 heading_class_label, heading_scores, heading_residual_label, heading_residuals_normalized,
                                 size_class_label, size_score, size_residual_label, size_residuals_normalized, 
                                 sem_class_label, sem_class_score,
                                 config):
    objectness_label = K.cast(objectness_label, 'float32')
    # some basis constant and parameters
    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    mean_size_arr = config.mean_size_arr
    batch_size = K.shape(object_assignment)[0]
    _, num_seed = K.int_shape(object_assignment)
    # -----------center loss---------------
    gt_center = center_label[:,:,:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)
    centroid_reg_loss1 = K.sum(dist1 * K.cast(objectness_label, 'float32'))/(K.sum(K.cast(objectness_label, 'float32'))+K.epsilon())
    # distance from each prediction to nearest gt center. make sure each prediction is accurate (high precision)
    centroid_reg_loss2 = K.sum(dist2 * K.cast(box_label_mask, 'float32'))/(K.sum(K.cast(box_label_mask,'float32'))+K.epsilon())
    # distance from each gt box center to nearst prediction. make sure each gt box has corresponding prediction (high recall)
    # Dataset gives fixed number of gt boxes. Some of them are zeros. use box label mask to remove them
    center_loss = centroid_reg_loss1 + centroid_reg_loss2
    
    # ----------heading loss---------------
    heading_class_label = custom_gather(heading_class_label, object_assignment) # select corresponding point as groundtruth B,K1 from B,K2
    heading_class_loss = softmax_crossentropy(heading_scores, heading_class_label, objectness_label)

    heading_residual_label = custom_gather(heading_residual_label, object_assignment) # select B,K1 from B,K2
    heading_residuals_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    heading_label_one_hot = K.one_hot(K.cast(heading_class_label,'int32'), num_heading_bin) # B, K1, num_class
    heading_residuals_normalized_loss = huber_loss(K.sum(heading_residuals_normalized * heading_label_one_hot, -1) -heading_residuals_normalized_label, delta=1.0) # B, K1
    heading_residuals_normalized_loss = K.sum(heading_residuals_normalized_loss*K.cast(objectness_label,'float32'))/(K.cast(K.sum(objectness_label),'float32')+K.epsilon())

    #-----------size loss-------------------
    size_class_label = custom_gather(size_class_label, object_assignment) # select B,K1 from B,K2. select GT for each prediction
    size_class_loss = softmax_crossentropy(size_score, size_class_label, objectness_label)

    object_assignment_expanded = K.tile(K.expand_dims(object_assignment,-1),(1,1,3))
    size_residual_label = custom_gather_v2(size_residual_label, object_assignment_expanded) # select B,K1,3 from B,K2,3
    size_label_one_hot = K.one_hot(K.cast(size_class_label,'int32'), num_size_cluster) # B, K1, sum_size_cluster
    size_label_one_hot_tiled = K.tile(K.expand_dims(K.cast(size_label_one_hot, 'float32'),axis=-1), (1,1,1,3)) # B, K1, sum_size_cluster, 3
    predicted_size_residual_normalized = K.sum(size_residuals_normalized * size_label_one_hot_tiled, 2) # B, K1, 3. choose prediction in one cluster

    mean_size_arr_expand = K.expand_dims(K.expand_dims(K.constant(mean_size_arr.astype(np.float32)), 0), 0) # (1, 1, num_size_cluster, 3)
    mean_size_label = K.sum(size_label_one_hot_tiled * K.tile(mean_size_arr_expand, (batch_size,num_seed,1,1)), 2) # (B,K,3)
    size_residuals_label_normalized = size_residual_label / mean_size_label # (B, K, 3)
    size_residuals_normalized_loss = K.mean(huber_loss(predicted_size_residual_normalized - size_residuals_label_normalized, delta=1.0), -1) # B,K,3 -> B,K
    size_residuals_normalized_loss = K.sum(size_residuals_normalized_loss * objectness_label)/ (K.sum(objectness_label)+K.epsilon())

    # -----------semantic class loss ------------
    sem_class_label = custom_gather(sem_class_label, object_assignment) # select B,K1 from B,K2
    sem_class_loss = softmax_crossentropy(sem_class_score, sem_class_label, objectness_label)

    return center_loss, heading_class_loss, heading_residuals_normalized_loss, size_class_loss, size_residuals_normalized_loss, sem_class_loss


def votenet_loss(args, config):
    '''
    for calculate the loss, following are needed. Be careful with the order!

    Args:
    Labels:
        0 - center_label,
        1 - heading_class_label, 
        2 - heading_residual_label,
        3 - size_class_label, 
        4 - size_residual_label,
        5 - sem_cls_label,
        6 - box_label_mask,
        7 - vote_label, 
        8 - vote_label_mask

    Predictions:
        9 - center,
        10 - heading_scores, 
        11 - heading_residuals_normalized,
        12 - size_scores, 
        13 - size_residuals_normalized,
        14 - sem_cls_scores,
        15 - seed_xyz, 
        16 - seed_inds,
        17 - vote_xyz,
        18 - objectness_score,
        19 - proposal_xyz

    Returns:
    0 - loss
    1 - vote_loss
    2 - objectness_loss
    3 - center_loss
    4 - heading_class_loss
    5 - heading_reg_loss
    6 - size_class_loss
    7 - size_reg_loss
    8 - sem_class_loss
    9 - box_loss
    '''
    # first unpack inputs
    # labels
    center_label = args[0]
    heading_class_label = args[1]
    heading_residual_label = args[2]
    size_class_label = args[3]
    size_residual_label = args[4]
    sem_cls_label = args[5]
    box_label_mask = args[6]
    vote_label = args[7]
    vote_label_mask = args[8]

    # predictions and sample points
    center = args[9]
    heading_scores = args[10]
    heading_residuals_normalized = args[11]
    size_scores = args[12]
    size_residuals_normalized = args[13]
    sem_cls_scores = args[14]
    seed_xyz = args[15]
    seed_inds = args[16]
    vote_xyz = args[17]
    objectness_score = args[18]
    proposal_xyz = args[19]
    
    # calculate the vote loss
    num_votes = K.int_shape(vote_xyz)[1]
    vote_xyz = K.reshape(vote_xyz, (-1,num_votes,3))
    vote_loss = compute_vote_loss(seed_xyz, vote_xyz, seed_inds, vote_label, vote_label_mask)
    # calculate the objectness loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(proposal_xyz, center_label, objectness_score)
    # calculate box and semantic classification loss
    box_and_sem_cls_loss = compute_box_and_sem_cls_loss(object_assignment, 
                                                        center, center_label, box_label_mask, 
                                                        objectness_label, heading_class_label, 
                                                        heading_scores, heading_residual_label, 
                                                        heading_residuals_normalized, size_class_label,
                                                        size_scores, size_residual_label, 
                                                        size_residuals_normalized, sem_cls_label, 
                                                        sem_cls_scores, config)
    # unpack the loss
    center_loss, heading_class_loss, heading_reg_loss, size_class_loss, size_reg_loss, sem_class_loss = box_and_sem_cls_loss
    # weighted sum
    box_loss = center_loss + 0.1*heading_class_loss + heading_reg_loss + 0.1*size_class_loss + size_reg_loss
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_class_loss
    loss = loss * 10

    return K.concatenate([K.reshape(loss,(1,)), 
            K.reshape(vote_loss,(1,)), 
            K.reshape(objectness_loss,(1,)),
            K.reshape(center_loss,(1,)),
            K.reshape(heading_class_loss,(1,)), 
            K.reshape(heading_reg_loss,(1,)),
            K.reshape(size_class_loss,(1,)),
            K.reshape(size_reg_loss,(1,)), 
            K.reshape(sem_class_loss,(1,)), 
            K.reshape(box_loss,(1,))], axis=-1)

def custom_gather_v2(reference, indices):
    '''
    only works for 3d
    gather with Pytorch behaviour(aixs = 1)
    '''
    indices = K.cast(indices, 'int32')
    assert len(K.int_shape(reference))==len(K.int_shape(indices))
    B = tf.shape(reference)[0]
    N = tf.shape(indices)[1]
    M = tf.shape(indices)[2]
    ii,_,kk = tf.meshgrid(tf.range(B),
                       tf.range(N), 
                       tf.range(M), indexing='ij')
    index = tf.stack([ii, indices, kk], axis=-1)
    result = tf.gather_nd(reference, index)
    return result

def custom_gather(reference, indices):
    '''
    only works for 2d
    gather with Pytorch behaviour(aixs = 1)
    '''
    # ref_shape = K.shape(reference)
    indices = K.cast(indices, 'int32')
    assert len(K.int_shape(reference))==len(K.int_shape(indices))
    B = tf.shape(reference)[0]
    N = tf.shape(indices)[1]
    ii,_ = tf.meshgrid(tf.range(B),
                       tf.range(N), indexing='ij')
    index = tf.stack([ii, indices], axis=-1)
    result = tf.gather_nd(reference, index)
    return result

def softmax_crossentropy(score, class_label, musk):
    '''
    Args:
        score: B, N, num_class
        class_label: B, N - not one hot, but sparse
        musk: B, N - filters out some loss
    Returns:
        loss: scalar
    '''
    score_softmax = K.softmax(score, axis=-1) # apply softmax
    class_label_onehot = K.one_hot(K.cast(class_label, "int32"), num_classes=K.int_shape(score)[-1]) # int to one hot
    class_label_onehot = K.cast(class_label_onehot, "float32") # cast type
    class_loss = K.categorical_crossentropy(class_label_onehot, score_softmax) # (B, K1)
    class_loss = K.sum(class_loss*K.cast(musk,'float32')) / (K.sum(K.cast(musk, 'float32'))+K.epsilon())
    return class_loss

def check_shape(tensor, name):
    print('[', name ,']',K.int_shape(tensor))

if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.array([[0, 1], [2, 1]], dtype=np.int32)
    x = tf.placeholder(tf.float32, shape=(None, 3))
    y = tf.placeholder(tf.int32, shape=(None, 2))

    gathered = custom_gather(x, y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(gathered, feed_dict={x:a, y:b}))