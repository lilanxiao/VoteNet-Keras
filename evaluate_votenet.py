import numpy as np
import os
import open3d as o3d
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
from votenet import create_votenet_body, create_votenet_inferencing
from dataset_model import SunrgbdDatasetConfig
from SUNRGBDDataset import SunrgbdDetectionVotesDataset
from ap_helper import parse_groundtruths, parse_predictions,flip_axis_to_depth,APCalculator
from plot_helper import create_bbox, create_pointcloud

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str, default='logs/votenet', help='Path to load the pretrained checkpoints [default logs/votenet]')
parser.add_argument('--checkpoint',type=str,default=None, help='File name of the check point used for inferencing. Load last check point in log_dir if not specified [default None]')
parser.add_argument('--num_points', type=int, default=20000, help='Number of input points [default: 20000]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Numbers of votes each seed generates [default: 1]')
parser.add_argument('--num_proposals',type=int, default=256, help='Number of proposals [default: 256]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Confidence threshhold for NMS [default 0.05]')
parser.add_argument('--nms_iou', type=float, default=0.25, help='IoU threshhold for NMS [default 0.25]')
parser.add_argument('--ap_iou', type=float, default=0.25, help='IoU threshhold for calculating the AP [default 0.25]')
parser.add_argument('--thresh_viz', type=float, default=0.5, help='Confidence threshold for visualization [default 0.5')
parser.add_argument('--no_viz', action='store_true', help='NOT use the visualization [default False]')
parser.add_argument('--use_color',action='store_true',help='Use RGB color as features')
parser.add_argument('--no_height',action='store_true',help='Do NOT use height as features')
parser.add_argument('--use_v1',action='store_true',help='Use v1 labels for SUN RGB-D dataset')
parser.add_argument('--random_proposal', action='store_true', help='Use random sampling instead of FPS in proposal module [default False]')
parser.add_argument('--offset',type=int, default=0, help='Offset value from the last check point [default 0]')
flags = parser.parse_args()

# make sure these parameters match the trained model
log_dir = flags.log_dir
checkpoint = flags.checkpoint
batch_size = flags.batch_size
num_points = flags.num_points
vote_factor = flags.vote_factor
use_color = flags.use_color
use_height = not flags.no_height
use_v1 = flags.use_v1
num_proposals = flags.num_proposals
random_proposal = flags.random_proposal
conf_thresh = flags.conf_thresh # threshhold of objectness
nms_iou = flags.nms_iou
ap_iou = flags.ap_iou
visualize_first_batch = not flags.no_viz
thresh_viz = flags.thresh_viz
offset = flags.offset
# --------------------------------------------------
shuffle_generator = True
per_class_proposal = True
cls_nms = True
num_features = 1*int(use_height) + 3*int(use_color)
DC = SunrgbdDatasetConfig()

net = create_votenet_inferencing(num_points=num_points, 
                                pcd_feature_dims=num_features, 
                                vote_factor=vote_factor, 
                                num_class=DC.num_class, 
                                num_head_bin=DC.num_heading_bin,
                                num_size_cluster=DC.num_size_cluster,
                                num_proposal=num_proposals, 
                                mean_size_arr=DC.mean_size_arr,
                                random_proposal=random_proposal,
                                config=DC)

dataset = SunrgbdDetectionVotesDataset(split_set='val', 
                                        batch_size=batch_size, 
                                        num_points=num_points, 
                                        shuffle=shuffle_generator, 
                                        use_color=False,
                                        use_height=use_height,
                                        use_v1=use_v1,
                                        augment=False)

if checkpoint is not None:
    ckpt = checkpoint
else:
    ckpts = os.listdir(log_dir)
    ckpts = sorted((list(ckpts)), key= lambda x: x[2:5])
    ckpt = ckpts[-1-offset]
net.load_weights(os.path.join(log_dir, ckpt), skip_mismatch=True, by_name=True)
print("---------- creating inference model --------------")
print("Use check point:", os.path.join(log_dir, ckpt))
print("---------------- inferencing ---------------------")

ap_calculator = APCalculator(ap_iou_thresh=ap_iou, class2type_map=DC.class2type)

for i_batch in range(len(dataset)):
    # get one batch
    x = dataset[i_batch][0] # ignore the "fake label" from Dataset
    # unpack datas
    pcd = x[0]
    center_label = x[1]
    heading_class_label = x[2]
    heading_residual_label = x[3]
    size_class_label = x[4]
    size_residual_label = x[5]
    sem_cls_label = x[6]
    box_label_mask = x[7]
    vote_label = x[8]
    vote_label_mask = x[9]
    # feed point cloud to inference model
    y_pred = net.predict(pcd)
    # unpack predictions
    objectness_score_normalized_batch = y_pred[0].astype(np.float32) # B, num_proposal,2
    center_batch = y_pred[1].astype(np.float32) # B, num_proposal,3
    heading_batch = y_pred[2].astype(np.float32) # B, num_proposal
    size_batch = y_pred[3].astype(np.float32) # B, num_proposal,3
    sem_class_normalized_batch = y_pred[4].astype(np.float32) # B, num_proposal,num_class
    seeds_xyz_batch = y_pred[5].astype(np.float32) # B, num_seeds,3
    votes_xyz_batch = y_pred[6].astype(np.float32) # B, num_seeds*vote_factor,3

    batch_pred_map_cls = parse_predictions(objectness_score_normalized_batch, 
                                            center_batch, 
                                            heading_batch, 
                                            size_batch, 
                                            sem_class_normalized_batch,
                                            conf_thresh, 
                                            nms_iou,
                                            DC.num_class, 
                                            per_class_proposal=per_class_proposal, 
                                            cls_nms=cls_nms)
    batch_gt_map_cls = parse_groundtruths(center_label.astype(np.float32), 
                                        heading_class_label.astype(np.float32), 
                                        heading_residual_label.astype(np.float32), 
                                        size_class_label.astype(np.float32), 
                                        size_residual_label.astype(np.float32), 
                                        sem_cls_label.astype(np.float32), 
                                        box_label_mask.astype(np.float32), 
                                        DC)
    ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    if i_batch%20 == 19:
        print("Done",i_batch+1,'of', len(dataset)+1)
    
    # visualize the first batch
    if visualize_first_batch and i_batch == 0:
        for i in range(batch_size):
            # take one point cloud
            points_xyz = pcd[i,:,:3]
            objectness_score_normalized = objectness_score_normalized_batch[i]
            center = center_batch[i]
            heading= heading_batch[i]
            size = size_batch[i]
            sem_class_class_normalized = sem_class_normalized_batch[i]
            seeds_xyz = seeds_xyz_batch[i]
            votes_xyz = votes_xyz_batch[i]
            # ----------------visualize point clouds, votes, and seeds, centers-----------------------------
            # create Open3d instance for visualization
            points_xyz_pcd = create_pointcloud(points_xyz,[0.5,0.5,0.5]) # grey
            seeds_xyz_pcd = create_pointcloud(seeds_xyz, [1,0,0]) # red
            votes_xyz_pcd = create_pointcloud(votes_xyz, [0,1,0]) # green
            objectness_musk = np.greater_equal(objectness_score_normalized[:,1],conf_thresh)
            center_p = center[objectness_musk,:]
            center_p_pcd = create_pointcloud(center_p,[0,0,1]) # blue
            draw_list = [seeds_xyz_pcd,points_xyz_pcd,votes_xyz_pcd,center_p_pcd]
            # draw_list = [points_xyz_pcd]
            # ----------------visualize bounding box --------------------------
            gt_bboxes = batch_gt_map_cls[i]
            # draw_list.append(create_bbox(x[1]) for x in gt_bboxes)
            for x in gt_bboxes:
                fliped_box = flip_axis_to_depth(x[1]) # flip back to world coordinate
                draw_list.append(create_bbox(fliped_box, [0,1,0])) # green box - GT
            
            pred_bbox = batch_pred_map_cls[i]
            for x in pred_bbox:
                if x[2] > thresh_viz: # obejctness threshhold for viusualisation
                    fliped_box = flip_axis_to_depth(x[1])
                    draw_list.append(create_bbox(fliped_box, [1,0,0])) # red box -prediction
            o3d.visualization.draw_geometries(draw_list)

print('----------calculating AP and AR------------------')
metric_dict = ap_calculator.compute_metrics()
print('With threshhold of %.2f 3DIoU :'% ap_iou)
for key in metric_dict:
    print('eval %s: %f'%(key, metric_dict[key]))