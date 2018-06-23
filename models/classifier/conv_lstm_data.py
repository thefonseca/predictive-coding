from keras.utils import Sequence, to_categorical
from keras.preprocessing import image

import glob
import os
import numpy as np
import bz2
import pickle as pkl
#from imageio import imread

import utils

'''
Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
'''

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=16, shuffle=True, fn_preprocess=None,
                 index_start=0, max_per_class=None, rescale=None,
                 seq_length=None, sample_step=1, target_size=None):
        'Initialization'
        self.batch_size = batch_size
        self.X = []
        self.y = []
        self.shuffle = shuffle
        self.index_start = index_start
        self.max_per_class = max_per_class
        self.seq_length = seq_length
        self.sample_step = sample_step
        self.rescale = rescale
        self.target_size = target_size
        self.fn_preprocess = fn_preprocess
        
    def flow_from_directory(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted(os.walk(data_dir).next()[1])
        data_pattern = '{}/*'
        
        for i, c in enumerate(self.classes):
            class_samples = sorted(glob.glob(os.path.join(data_dir, data_pattern.format(c))))
            self.__process_class_samples(i, class_samples, class_samples)
            
        self.__postprocess()
        return self
        
    def flow(self, X, y, sources=None):
        self.classes = sorted(list(set(y)))
        self.sources = sources
        
        for i, c in enumerate(self.classes):
            sample_indices = [k for k, y_ in enumerate(y) if y_ == c]
            class_samples = X[sample_indices]
            class_sources = sources[sample_indices]
            self.__process_class_samples(i, class_samples, class_sources)
            
        self.__postprocess()
        return self
    
    def __postprocess(self):
        self.n_classes = len(self.classes)
        self.data_shape = self.__load_data(0).shape
        msg = 'Found {} samples belonging to {} classes in {}'
        print(msg.format(len(self.X), self.n_classes, self.data_dir))
        self.on_epoch_end()
        
    def __process_class_samples(self, class_index, class_samples, class_sources=None):
        if self.max_per_class is None or self.index_start + self.max_per_class >= 0:
            class_samples = class_samples[self.index_start::self.sample_step]
        else:
            index_end = self.index_start + self.max_per_class
            class_samples = class_samples[self.index_start:index_end:self.sample_step]

        if self.seq_length:
            class_samples = self.__to_sequence(class_samples, class_sources)

        self.y.extend([class_index] * len(class_samples))
        self.X.extend(class_samples)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
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
            y[i] = self.y[index]
            
        return X, to_categorical(y, num_classes=self.n_classes)
    
    def __preprocess(self, img):
        '''if self.target_size:
            img = utils.resize_img(img, self.target_size)
        
        if self.rescale:
            img = self.rescale * img'''
        if self.fn_preprocess:
            img = self.fn_preprocess(img)
        return img
    
    def __load_image(self, filename):
        img = image.load_img(filename, target_size=self.target_size)
        img = image.img_to_array(img)
        #img = np.expand_dims(img, axis=0)
        #img = imread(filename)
        return self.__preprocess(img)
    
    def __load_pickle(self, filename):
        with bz2.BZ2File(filename, 'rb') as f:
            return pkl.load(f)
        
    def __load_sample(self, filename):
        if filename.lower().endswith('.pkl'):
            sample = self.__load_pickle(filename)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample = self.__load_image(filename)
        return sample
    
    def __load_seq_data(self, index):
        seq = self.X[index]
        seq_data = []
        
        for sample in seq:
            seq_data.append(self.__load_sample(sample))
                        
        return np.array(seq_data)
    
    def __load_data(self, index):
        if self.seq_length:
            data = self.__load_seq_data(index)
        else:
            data = self.__load_sample(self.X[index])
            
        return data
    
    def __to_sequence(self, samples, sources):
        seq = []
        prev_source = None
        seqs = []

        for i, s in enumerate(sources):
            source = '__'.join(s.split('__')[:-1]) # NAME_OF__SOURCE__frame_001.pkl => NAME_OF__SOURCE

            if prev_source is None or prev_source == source:
                seq.append(samples[i])
            else:
                if len(seq) >= self.seq_length:
                    seqs.append(seq[:self.seq_length])
                    
                seq = [s]

            prev_source = source
            
        return seqs