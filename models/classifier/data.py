from keras.utils import Sequence, to_categorical
from keras.preprocessing import image
from keras import backend as K

import glob
import os
import numpy as np
import pickle as pkl
import random as rn

# Getting reproducible results:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

'''
Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
'''

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=16, shuffle=False, fn_preprocess=None,
                 index_start=0, max_per_class=None, seq_length=None, 
                 sample_step=1, seq_overlap=0, target_size=None, 
                 classes=None, data_format=K.image_data_format(), 
                 output_mode=None, rescale=None, max_seq_per_source=None, 
                 min_seq_length=1, pad_sequences=False, return_sources=False):
        
        'Initialization'
        self.batch_size = batch_size
        self.X = []
        self.y = []
        self.shuffle = shuffle
        self.index_start = index_start
        self.max_per_class = max_per_class
        self.seq_length = seq_length
        self.min_seq_length = min_seq_length
        self.sample_step = sample_step
        self.target_size = target_size
        self.fn_preprocess = fn_preprocess
        self.return_sources = return_sources
        self.classes = classes
        self.data_format = data_format
        self.output_mode = output_mode
        self.seq_overlap = seq_overlap
        self.rescale = rescale
        self.max_seq_per_source = max_seq_per_source
        self.source_count = {}
        self.pad_sequences = pad_sequences
        
    def flow_from_directory(self, data_dir):
        self.data_dir = data_dir
        
        if self.classes is None:
            self.classes = sorted(next(os.walk(data_dir))[1])
        data_pattern = '{}/*'
        
        total_samples = 0
        for i, c in enumerate(self.classes):
            class_samples = sorted(glob.glob(os.path.join(data_dir, data_pattern.format(c))))
            total_samples += self.__process_class_samples(i, class_samples, class_samples)
        
        msg = 'Found {} samples belonging to {} classes in {}'
        print(msg.format(total_samples, len(self.classes), self.data_dir))

        self.__postprocess()
        return self
        
    def flow(self, X, y, sources=None):
        if self.classes is None:
            self.classes = sorted(list(set(y)))
        self.sources = sources
        
        total_samples = 0
        for i, c in enumerate(self.classes):
            sample_indices = [k for k, y_ in enumerate(y) if y_ == c]
            class_samples = X[sample_indices]
            class_sources = None
            if sources:
                class_sources = sources[sample_indices]
            total_samples += self.__process_class_samples(i, class_samples, class_sources)
            
        msg = 'Found {} samples belonging to {} classes'
        print(msg.format(total_samples, len(self.classes)))
        
        self.__postprocess()
        return self
    
    def __process_class_samples(self, class_index, class_samples, class_sources=None):
        if 0 < self.index_start < 1:
            index_start = int(self.index_start * len(class_samples))
        else:
            index_start = self.index_start
        
        if self.max_per_class is None or (index_start < 0 <= index_start + self.max_per_class):
            class_samples = class_samples[index_start::self.sample_step]
            class_sources = class_sources[index_start::self.sample_step]
        else:
            if 0 < self.max_per_class < 1:
                index_end = index_start + int(self.max_per_class * len(class_samples))
            else:
                index_end = index_start + self.max_per_class
            class_samples = class_samples[index_start:index_end:self.sample_step]
            class_sources = class_sources[index_start:index_end:self.sample_step]
        
        total_class_samples = len(class_samples)
        if self.seq_length:
            class_samples = self.__to_sequence(class_samples, class_sources)
        
        self.y.extend([class_index] * len(class_samples))
        self.X.extend(class_samples)
        return total_class_samples
        
    def __postprocess(self):
        self.n_classes = len(self.classes)
        if len(self.X) == 0:
            print('No data found in {}!'.format(self.data_dir))
        else:
            self.sources = []
            
            if self.seq_length:
                # Check sequence counts consistency
                for k, c in self.source_count.items():
                    if self.max_seq_per_source and c > self.max_seq_per_source:
                        raise ValueError('A sequence exceeds the maximum length')
                
                # Get sequence length distribution
                seq_length_dist = {}
                for seq in self.X:
                    count_per_length = seq_length_dist.get(len(seq), 0)
                    seq_length_dist[len(seq)] = count_per_length + 1
                    
                    # get sources (e.g. videos)
                    for sample in seq:
                        source = '__'.join(sample.split('__')[:-1])
                        if len(source) > 0 and source != 'padding':
                            self.sources.append(source)
                
                msg = 'Found {} sequences belonging to {} classes'
                print(msg.format(len(self.X), self.n_classes))
                
                print('Sequence distribution:')
                total = 0
                for length, count in sorted(seq_length_dist.items()):
                    print('- {} sequences of length {}'.format(count, length))
                    total += count * length
                all_samples = [s for seq in self.X for s in seq if s != 'padding']
                unique_samples = len(set(all_samples))
                print('Total samples used: {}'.format(unique_samples))
                
            else:
                for x in self.X:
                    source = '__'.join(x.split('__')[:-1])
                    if len(source) > 0 and source != 'padding':
                        self.sources.append(source)
                
            self.sources = sorted(list(set(self.sources)))
            print('Total sources used: {}'.format(len(self.sources)))
            
            self.data_shape = self.__load_data(0).shape
            print('Data shape: {}'.format(self.data_shape))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        return self.__data_generation(indexes)

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
        sources = []
        
        # Generate data
        for i, index in enumerate(indexes):
            X[i,] = self.__load_data(index)
            
            # Store class
            y[i] = self.y[index]
            sources.append(self.X[index])
            
        if self.data_format == 'channels_first':
            X = np.transpose(X, (0, 1, 4, 2, 3))
         
        if self.output_mode is not None and self.output_mode == 'error':  
            data = (X, np.zeros(self.batch_size, np.float32))
        elif self.output_mode is not None and self.output_mode == 'category_and_error':
            data = (X, {
              'category_prediction': to_categorical(y, num_classes=self.n_classes), 
              'prednet_error': np.zeros(self.batch_size, np.float32)
              })
        else:
            data = (X, to_categorical(y, num_classes=self.n_classes))
        
        if self.return_sources:
            data += (np.array(sources),)
        
        return data
    
    def __preprocess(self, img):
        '''if self.target_size:
            img = utils.resize_img(img, self.target_size)'''
        if self.rescale:
            img = self.rescale * img
        if self.fn_preprocess:
            img = self.fn_preprocess(img)
        return img
    
    def __load_image(self, filename):
        img = image.load_img(filename, target_size=self.target_size)
        img = image.img_to_array(img)
        # img = imread(filename)
        return self.__preprocess(img)
    
    def __load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)
        
    def __load_sample(self, filename):
        if filename.lower().endswith('.pkl'):
            sample = self.__load_pickle(filename)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample = self.__load_image(filename)
        elif filename == 'padding':
            sample = np.zeros(self.sample_shape)
        else:
            raise ValueError('{} format is not supported'.format(filename))
            
        self.sample_shape = sample.shape
        return sample
    
    def __load_seq_data(self, index):
        seq = self.X[index]
        seq_data = []
        
        for sample in seq:
            seq_data.append(self.__load_sample(sample))
        
        return np.array(seq_data)
    
    def __load_data(self, index):
        if len(self.X) <= index:
            return None
        
        if self.seq_length:
            data = self.__load_seq_data(index)
        else:
            data = self.__load_sample(self.X[index])
            
        return data
    
    def __add_incomplete_sequence(self, seq, source_seq, seqs, source_seqs):
        if self.pad_sequences:
            padding_item = 'padding' # np.zeros_like(seq[-1])
            seq.extend([padding_item] * (self.seq_length - len(seq)))
            source_seq.extend([padding_item] * (self.seq_length - len(seq)))
        seqs.append(seq)
        source_seqs.append(source_seq)
    
    def __to_sequence(self, samples, sources):
        seqs = []
        source_seqs = []
        
        i = 0
        while i < len(sources) - self.seq_overlap - 1:
            seq = []
            source_seq = []
            prev_source = None
            
            # Try to get one sequence of length self.seq_length
            for j in range(i, i + self.seq_length):
                # NAME_OF__SOURCE__frame_001.pkl => NAME_OF__SOURCE
                source = '__'.join(sources[j].split('__')[:-1]) 
                # print(i, j, source)
                
                count = self.source_count.get(source, 0)
                
                if self.max_seq_per_source and count >= self.max_seq_per_source:
                    # print('count:', count, samples[j])
                    i = j + 1
                    break
                
                if prev_source is None or prev_source == source:
                    seq.append(samples[j])
                    source_seq.append(source)
                    
                    # print('prev_source == source:', i, j)
                    if len(seq) == self.seq_length:
                        # print('added:', i, j, source)
                        seqs.append(seq)
                        source_seqs.append(source_seq)
                        i = j - self.seq_overlap + 1
                        self.source_count[source] = count + 1
                        break
                else:
                    if self.min_seq_length <= len(seq):
                        self.__add_incomplete_sequence(seq, source_seq, 
                                                       seqs, source_seqs)
                    i = j
                    break
                    
                prev_source = source
                # print(len(sources), self.seq_overlap)
                if j == len(sources) - 1:
                    if self.min_seq_length <= len(seq):
                        self.__add_incomplete_sequence(seq, source_seq, 
                                                       seqs, source_seqs)
                    i = j
                    break
            
        return seqs
