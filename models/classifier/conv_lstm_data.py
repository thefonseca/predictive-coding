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
                 index_start=0, max_per_class=None, seq_length=5):
        'Initialization'
        self.batch_size = batch_size
        self.labels = []
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.index_start = index_start
        self.max_per_class = max_per_class
        self.seq_length = seq_length
        
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
            
            seq = []
            prev_source = None
            class_seqs = []
            
            for item in class_items:
                source = '__'.join(item.split('__')[:-1]) # NAME_OF__SOURCE__frame_001.pkl => NAME_OF__SOURCE
                #print(len(seq), source)
                if prev_source is None or prev_source == source:
                    seq.append(item)
                else:
                    if len(seq) >= seq_length:
                        class_seqs.append(seq[:seq_length])
                    
                    seq = [item]
                    
                prev_source = source
                
            
            self.labels.extend([i] * len(class_seqs))
            self.data_list.extend(class_seqs)
            
        self.data_shape = self.__load_data(0).shape
        
        msg = 'Found {} samples belonging to {} classes in {}'
        print(msg.format(len(self.data_list), len(self.classes), data_dir))
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
        seq = self.data_list[index]
        seq_data = []
        for item in seq:
            with bz2.BZ2File(item, 'rb') as f:
                 seq_data.append(pkl.load(f))
                    
        return np.array(seq_data)