from keras.utils import Sequence, to_categorical
import glob
import os
import numpy as np
import bz2
import pickle as pkl

'''
Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
'''

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, data_dir, batch_size=16, shuffle=True, 
                 index_start=0, max_per_class=None):
        'Initialization'
        self.batch_size = batch_size
        self.labels = []
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.index_start = index_start
        self.max_per_class = max_per_class
        
        self.classes = sorted(os.walk(data_dir).next()[1])
        self.n_classes = len(self.classes)
        data_pattern = '{}/*'
        self.data_list = []
        
        for i, c in enumerate(self.classes):
            class_items = sorted(glob.glob(os.path.join(data_dir, data_pattern.format(c))))
            
            if max_per_class is None:
                class_items = class_items[index_start:]
            else:
                class_items = class_items[index_start:index_start+max_per_class]
            
            self.labels.extend([i] * len(class_items))
            self.data_list.extend(class_items)
            
        self.data_shape = self.__load_data(0).shape
            
        print('Found {} samples belonging to {} classes'.format(len(self.data_list), 
                                                      len(self.classes)))
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *shape)
        # Initialization
        X = np.empty((self.batch_size,) + self.data_shape)
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
        for i, index in enumerate(indexes):
            X[i,] = self.__load_data(index)
            
            # Store class
            y[i] = self.labels[index]
            
        return X, to_categorical(y, num_classes=self.n_classes)
    
    def __load_data(self, index):
        with bz2.BZ2File(self.data_list[index], 'rb') as f:
            return pkl.load(f)