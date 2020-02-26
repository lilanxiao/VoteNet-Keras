import keras
import json
import os
import os.path
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider

class ModelNet40Dataset(keras.utils.Sequence):
    def __init__(self, root, batch_size=16, npoints = 1024, split = 'train', shuffle=True, augment=True):
        '''
        root: file path of data
        batch size:
        n_points: number of points
        split: 'train' or 'test'
        shuffle: if true, shuffle the dataset
        augment: if true, do data augmentation
        '''
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.split = split
        self.augment = augment
        if split is None:
            if split == 'train': self.shuffle = True
            else : self.shuffle == False 
        else:
            self.shuffle = shuffle
        # load category names
        self.catfile = os.path.join(self.root, 'shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        shape_ids = {}
        # load data path
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train_files.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test_files.txt'))]
        # it's a small dataset, load all data in memory
        datas = [provider.load_h5(x)[0] for x in shape_ids[split]]
        labels = [provider.load_h5(x)[1] for x in shape_ids[split]]
        self.datas = np.concatenate(datas, axis = 0) # concatenate list to numpy array
        self.labels = np.concatenate(labels, axis = 0)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.datas)/self.batch_size)) # length is the max batch numbers

    def on_epoch_end(self):
        '''
        on epoche end, shuffle the datatset
        '''
        self.indices = np.arange(len(self.datas)) # save the current indices
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_idx):
        x = np.zeros((self.batch_size, self.npoints, 3))
        y = np.zeros((self.batch_size,))
        for i, idx in enumerate(batch_idx, 0):
            x[i] = self.datas[idx, 0:self.npoints, :] # take the first n points. TODO: random choice
            y[i] = self.labels[idx]
        if self.augment: # and np.random.rand()>0.5:
            # implement data augmentation to the whole BATCH
            rotated_x = provider.rotate_point_cloud(x) # rotate around x-axis
            rotated_x = provider.rotate_perturbation_point_cloud(rotated_x) # slightly rotate around every aixs
            jittered_x = provider.random_scale_point_cloud(rotated_x) # random scale a little bit
            jittered_x = provider.shift_point_cloud(jittered_x) # shift a little
            jittered_x = provider.jitter_point_cloud(jittered_x) # add random noise (jitter)
            jittered_x = provider.shuffle_points(jittered_x) # shuffle the point. for FPS
            x = jittered_x
        return x, keras.utils.to_categorical(y, num_classes=len(self.cat))

    def __getitem__(self, index):
        '''
        get one batch
        '''
        batch_idx = self.indices[index * self.batch_size: (index+1)*self.batch_size]
        x, y = self.__data_generation(batch_idx)
        return x,y

if __name__ == "__main__":
    root = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048')
    dataset = ModelNet40Dataset(root=root, batch_size=8, npoints=2048, split='train', shuffle=True)
    print('batch number:', len(dataset))
    x,y = dataset[1]
    print(x.shape)
    print(y.shape)
