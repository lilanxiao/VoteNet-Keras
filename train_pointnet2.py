'''
Training script for pointnet++-SSG and VoteNet-Backbone
used to validate the Keras implementation of some custome layers and operations

Lanxiao Li 2020
'''
import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
from modelnet40dataset import ModelNet40Dataset
from pointnet2 import pointnet2_cls_ssg
from keras import optimizers
from callbacks import Step, Divide_lr, BNDecayScheduler
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from backbone import make_classifier

BASE_DIR = os.path.abspath(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', default=2048, help='Number of input points [default: 2048]')
parser.add_argument('--batch_size', default=16, help='Batch size [default: 16]')
parser.add_argument('--epochs', default=150, help='Epochs to train [default: 150]')
parser.add_argument('--log_dir',default='logs/pointnet2', help='Path to save the checkpoints [default logs/pointnet2]')
parser.add_argument('--lr', default=0.001, help='Initail learning rate [default 0.001]')
parser.add_argument('--data_root', default='data/modelnet40_ply_hdf5_2048', help='Path to the training [default data/modelnet40_ply_hdf5_2048]')
parser.add_argument('--model', default='pointnet2', help='Use pointnet2 or votenet_backbone as classifier [default pointnet2]')
parser.add_argument('--lr_divide_factor', default=1.5, help='Divide factor of learning rate [default 1.5]')
parser.add_argument('--lr_divide_interval', default=15, help='Epochs between every update of learning rate [default 15]')
parser.add_argument('--bn_momentum', default=0.5, help='Initial batch norm momentum [default 0.5]')
parser.add_argument('--bn_decay_rate', default=0.5, help='Decay rate of batch norm momentum [default 0.5')
parser.add_argument('--bn_decay_interval', default=15, help='Interval between updates of batch norm momentum [default 15]')
parser.add_argument('--bn_clip', default=0.99, help='Max value of batch norm momentum [default 0.99]')
FLAGS = parser.parse_args()

num_points = FLAGS.num_points
batch_size = FLAGS.batch_size
epochs = FLAGS.epochs
log_dir = FLAGS.log_dir
lr = FLAGS.lr
data_root = FLAGS.data_root
classifier_name = FLAGS.model # pointnet2 or votenet_backbone
lr_divide_factor = FLAGS.lr_divide_factor
lr_divide_interval = FLAGS.lr_divide_interval
bn_momentum_init = FLAGS.bn_momentum
bn_decay_rate = FLAGS.bn_decay_rate
bn_decay_interval = FLAGS.bn_decay_interval
bn_clip = FLAGS.bn_clip
#------------------------------------------------------------
num_class = 40
num_dim = 3
if not os.path.exists(log_dir): 
    os.makedirs(log_dir)
# Original implementation use intensive data augmentation to all training data. (89.4% val_acc)
# test shows that performing agumentation to only half of data delievers better performance (92.0% val_acc, 92.3 train_acc).
# Without any data augmentation, performance (91.9% val_acc) is also better than original one. But it suffers
# severe overfitting (~98% train_acc) at the same time. In conclusion, half data augumentation might be the best choice. 
# Perhaps as the network has already used strong dropout (two layers with 0.5 drop rate), strong data 
# augumentation harms the performance.
train_set = ModelNet40Dataset(root=data_root, batch_size=batch_size, npoints=num_points, split='train', shuffle=True, augment=True)
val_set = ModelNet40Dataset(root=data_root, batch_size=batch_size, npoints=num_points, split='test', shuffle=False, augment=False)
lr_scheduler = Step([20, 40, 60], [lr, lr/10, lr/100, lr/1000])
lr_divider = Divide_lr(15,1.5)
bn_decay_scheduler = BNDecayScheduler(bn_init=bn_momentum_init, decay_rate=bn_decay_rate, interval=bn_decay_interval, clip=bn_clip)
ckpt = ModelCheckpoint(os.path.join(log_dir,'ep{epoch:03d}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5'),
                       save_best_only=True,
                       monitor='val_acc',
                       save_weights_only=True,
                       period=1)
if classifier_name == 'pointnet2':
    model = pointnet2_cls_ssg(num_class, num_points, num_dim)
elif classifier_name == 'votenet_backbone':
    model = make_classifier(num_points, 0)
else:
    print("Error: Unknow Classfier:", classifier_name)
    exit()
adam = optimizers.Adam(lr)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
with open('report.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
plot_model(model, to_file='pointnet2.png', show_shapes=True,show_layer_names=True)
model.fit_generator(train_set, 
                    epochs=epochs,
                    validation_data=val_set,
                    callbacks=[lr_divider, ckpt, bn_decay_scheduler],
                    workers = 4, 
                    max_queue_size=400, 
                    use_multiprocessing=False, 
                    verbose=1)
