import numpy as np
import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
from dataset_model import SunrgbdDatasetConfig
from SUNRGBDDataset import SunrgbdDetectionVotesDataset
from votenet import create_votenet
from keras.optimizers import Adam
from callbacks import Step, BNDecayScheduler
from keras.callbacks import ModelCheckpoint
import keras.backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', default=20000, help='Number of input points [default: 20000]')
parser.add_argument('--batch_size', default=8, help='Batch size [default: 8]')
parser.add_argument('--vote_factor', default=1, help='Numbers of votes each seed generates [default: 1]')
parser.add_argument('--epochs', default=180, help='Epochs to train [default: 180]')
parser.add_argument('--num_proposals', default=256, help='Number of proposals [default: 256]')
parser.add_argument('--log_dir',default='logs/votenet', help='Path to save the checkpoints [default logs/votenet]')
parser.add_argument('--lr', default=0.001, help='Initail learning rate [default 0.001]')
parser.add_argument('--lr_decay_step', type=list, default=[80,120,160], help='When to decay the learning rate [default: [80,120,160]')
parser.add_argument('--lr_decay_factor', default=10, help='Learning rate decay factor [default 10]')
parser.add_argument('--bn_momentum', default=0.5, help='Initial batch norm momentum [default 0.5]')
parser.add_argument('--bn_decay_rate', default=0.5, help='Decay rate of batch norm momentum [default 0.5')
parser.add_argument('--bn_decay_interval', default=20, help='Interval between updates of batch norm momentum [default: 20]')
parser.add_argument('--bn_clip', default=0.999, help='Max value of batch norm momentum [default 0.999]')
parser.add_argument('--use_color',action='store_true',help='Use RGB color as features')
parser.add_argument('--no_height',action='store_true',help='Do NOT use height as features')
parser.add_argument('--use_v1',action='store_true',help='Use v2 labels for SUN RGB-D dataset')
parser.add_argument('--random_proposal', action='store_true', help='Use random sampling instead of FPS in proposal module [default False]')
flags = parser.parse_args()

log_dir = flags.log_dir
batch_size = flags.batch_size
num_points = flags.num_points
vote_factor = flags.vote_factor
num_proposals = flags.num_proposals
epochs = flags.epochs
lr = flags.lr
lr_decay_step = flags.lr_decay_step
lr_decay_factor = flags.lr_decay_factor
bn_momentum_init = flags.bn_momentum
bn_decay_rate = flags.bn_decay_rate
bn_decay_interval = flags.bn_decay_interval
bn_clip = flags.bn_clip
use_color = flags.use_color
use_height = not flags.no_height
use_v1 = flags.use_v1
random_proposal = flags.random_proposal
#----------------------------------------------------

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

num_features = 1*int(use_height) + 3*int(use_color)
Dataset_Config = SunrgbdDatasetConfig()
lr_decay_result = lr/np.power(lr_decay_factor, np.arange(len(lr_decay_step)+1))

train_set = SunrgbdDetectionVotesDataset(split_set='train', 
                                        batch_size=batch_size,
                                        num_points=num_points,
                                        shuffle=True,
                                        use_color=use_color,
                                        use_height=use_height,
                                        use_v1=use_v1,
                                        augment=True)
val_set = SunrgbdDetectionVotesDataset(split_set='val',
                                        batch_size=batch_size,
                                        num_points=num_points,
                                        shuffle=True,
                                        use_color=use_color,
                                        use_height=use_height,
                                        use_v1=use_v1,
                                        augment=False)
net = create_votenet(num_points=num_points, 
                    pcd_feature_dims=num_features,
                    vote_factor=vote_factor,
                    num_class=Dataset_Config.num_class,
                    num_head_bin=Dataset_Config.num_heading_bin,
                    num_size_cluster=Dataset_Config.num_size_cluster,
                    num_proposal=num_proposals,
                    mean_size_arr=Dataset_Config.mean_size_arr,
                    random_proposal = random_proposal,
                    config = Dataset_Config)
step = Step(lr_decay_step, lr_decay_result, 0)
bn_decay_scheduler = BNDecayScheduler(bn_init=bn_momentum_init, decay_rate=bn_decay_rate, interval=bn_decay_interval,clip=bn_clip)
ckpt = ModelCheckpoint(os.path.join(log_dir,'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                       save_best_only=False,
                       monitor='val_loss',
                       save_weights_only=True,
                       period=1)
net.summary()
def loss_components(idx,name):
    def choice(y_true, y_pred):
        return y_pred[idx]
    choice.__name__ = name
    return choice
net.compile(optimizer=Adam(lr), 
            loss={'votenet_loss': loss_components(0,'total')},
            metrics=[loss_components(1,'vote'),
            loss_components(2, 'obj'),
            loss_components(3, 'cent'),
            loss_components(4, 'h_cls'),
            loss_components(5, 'h_reg'),
            loss_components(6, 's_cls'),
            loss_components(7, 's_reg'),
            loss_components(8, 'sem')]) # use short name to avoid broken display
            # fake loss function and metrics here. Take the output of last Lambda layer as the true loss
            # view components of total loss as metrics to monitor the training.

net.fit_generator(train_set, 
                epochs=epochs, 
                validation_data=val_set,
                callbacks=[ckpt, step, bn_decay_scheduler], 
                max_queue_size=200,
                workers = 4, 
                use_multiprocessing=True, 
                verbose=1)
