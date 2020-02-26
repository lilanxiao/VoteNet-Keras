
## Introduction

This repository is an unofficial Keras/Tensorflow implementation of VoteNet. (Official GitHub [here](https://github.com/facebookresearch/votenet)).

## Environments

The code is tested with:

    Ubuntu 18.04
    Keras 2.2.0
    Tensorflow 1.10
    CUDA 9.2

*For some personal reason I have to develop with such old version Keras and Tensorflow

The following Python dependencies are also required:

    numpy
    scipy
    open3d # for visualization

## Compile Custom Tensorflow Operators

The VoteNet backbone uses some custom TF operators. The corresponding code is copied from [PointNet++](https://github.com/charlesq34/pointnet2) repo. To compile then, run following scripts.

    tf_ops/3d_interpolation/tf_interpolate_compile.sh
    tf_ops/grouping/tf_grouping_compile.sh
    tf_ops/sampling/tf_sampling_compile.sh

Configure `TF_ROOT` and `CUDA_ROOT` according to your own environment. If necessary, read instruction in PointNet++ repo for reference.

## Prepare the Training Data

This repo ONLY implements the SUN RGB-D version of VoteNet. Follow the instruction in the official VoteNet repo [here](https://github.com/facebookresearch/votenet/tree/master/sunrgbd) to prepare the training data. Move generated data and labels in folder `data`. The folder should have the following subfolders:

    sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_train
    sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_val
    sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_train
    sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_val
    sunrgbd/sunrgbd_trainval

*Optional: to train the Keras implementation of PointNet++, download ModelNet40 from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and move the data to folder `data`.

## Evaluate Pretrained Model

Run following script to evaluate VoteNet. The pretraind weight in folder `logs\votenet` will be loaded automatically. NOTE: this model is trained on label v2.

    python evaluate_votenet.py

The inferencing result of the first batch will be visualized via Open3d. To close the pop out window, press `ESC`. After that, the model will be evaluated on the whole validation set and the AP and AR will be printed.

Here is an example. The red and green boxes illustrate predicted and ground truth bounding box respectively. The green and blue points represent votes and high confidence centers.

![prediction_1](/images/prediction1.png)

To view more options, run `python evaluate_votenet.py -h`

    -h, --help            show this help message and exit
    --log_dir LOG_DIR     Path to load the pretrained checkpoints [default
                        logs/votenet]
    --checkpoint CHECKPOINT
                        File name of the check point used for inferencing.
                        Load last check point in log_dir if not specified
                        [default None]
    --num_points NUM_POINTS
                        Number of input points [default: 20000]
    --batch_size BATCH_SIZE
                        Batch size [default: 8]
    --vote_factor VOTE_FACTOR
                        Numbers of votes each seed generates [default: 1]
    --num_proposals NUM_PROPOSALS
                        Number of proposals [default: 256]
    --conf_thresh CONF_THRESH
                        Confidence threshhold for NMS [default 0.05]
    --nms_iou NMS_IOU     IoU threshhold for NMS [default 0.25]
    --ap_iou AP_IOU       IoU threshhold for calculating the AP [default 0.25]
    --thresh_viz THRESH_VIZ
                        Confidence threshold for visualization [default 0.5
    --no_viz              NOT use the visualization [default False]
    --use_color           Use RGB color as features
    --no_height           Do NOT use height as features
    --use_v1              Use v1 labels for SUN RGB-D dataset
    --random_proposal     Use random sampling instead of FPS in proposal module
                        [default False]

## Train from Scratch

Run following script to train VoteNet on SUN-RGB-D. The model needs about 12 hours (with multiprocessing in data generating) to train on a single Nvidia RTX 2080Ti. Since several loss values are displayed during the training, it's recommanded to maximize the terminal. Otherwise the display would be unstable and ugly.

    python train_votenet.py

Run following script to train PointNet++ on ModelNet40

    python train_poinet2.py

To check the hyperparameters and other options, run

    python train_votenet.py -h

or

    python train_pointnet2.py -h 

for more details.

## Citation

If you find our work useful in your research, please consider citing:

    @inproceedings{qi2019deep,
        author = {Qi, Charles R and Litany, Or and He, Kaiming and Guibas, Leonidas J},
        title = {Deep Hough Voting for 3D Object Detection in Point Clouds},
        booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
        year = {2019}
    }

    @article{qi2017pointnetplusplus,
      title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
      author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1706.02413},
      year={2017}
    }

## Known Issues

### 1.Performance Gap

The pretrained model can make decent predictions but the overall performance is obviously worse than the original implementation. Here is the result I get:

    With threshhold of 0.25 3DIoU :
    eval bed Average Precision: 0.847250
    eval table Average Precision: 0.460003
    eval sofa Average Precision: 0.645876
    eval chair Average Precision: 0.657584
    eval toilet Average Precision: 0.682671
    eval desk Average Precision: 0.206105
    eval dresser Average Precision: 0.225193
    eval night_stand Average Precision: 0.430959
    eval bookshelf Average Precision: 0.228098
    eval bathtub Average Precision: 0.646078
    eval mAP: 0.502982
    eval bed Recall: 0.951456
    eval table Recall: 0.861040
    eval sofa Recall: 0.904306
    eval chair Recall: 0.871016
    eval toilet Recall: 0.944828
    eval desk Recall: 0.801807
    eval dresser Recall: 0.766055
    eval night_stand Recall: 0.862745
    eval bookshelf Recall: 0.641844
    eval bathtub Recall: 0.857143
    eval AR: 0.846224

 By looking into the losses, the value of objectness loss and center loss is higher than they should be. There are probably some bugs in this code but I can't find them out.

### 2.Duplicate random number in date augmentation with multiprocessing

The SunrgbdDetectionVotesDataset gets duplicate random number when `use_multiprocessing=True`,  which makes the data augmentation not that "random" and might drop the performance. The model needs more time to train without multiprocessing. The usage of GPU is low especially in the validation phase.

    with multiprocessing: ~250 s/epoch
    without multiprocessing: ~320 s/epoch
    Hardware: Nvidia RTX 2080Ti, Intel i7-8700

This is a known issue in NumPy according to [here](https://github.com/pytorch/pytorch/issues/5059). Pytorch has `worker_init_fn` in Dataloader which allows resetting of random seed. But Keras doesn't has the equivalence.
