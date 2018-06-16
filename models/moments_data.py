import pickle as pkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
from imageio import imread
from skimage.transform import resize
import os

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, img_dir, source_file, nt, frame_step=1, seq_overlap=0,
                 batch_size=9, shuffle=False, seed=None, max_seq_per_video=None,
                 output_mode='error', N_seq=None, max_missing_frames = 10,
                 data_format=K.image_data_format(), img_size=(128, 160)):
        
        # source for each image so when creating sequences 
        # can assure that consecutive frames are from same video
        self.sources = pkl.load(open(source_file, "rb")) 
        
        self.nt = nt
        self.frame_step = frame_step
        self.batch_size = batch_size
        self.data_format = data_format
        
        #assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        default_output_modes = ['prediction', 'error', 'all', 'representation']
        nb_layers = 4 # TODO: add to parameters 
        layer_output_modes = [layer + str(n) for n in range(nb_layers) for layer in ['R', 'E', 'A', 'Ahat']]
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode
        assert (seq_overlap >= 0 and seq_overlap < self.nt), 'sequence overlap must be between >= 0 and < nt'
        self.seq_overlap = seq_overlap
        self.max_seq_per_video = max_seq_per_video
        
        self.img_dir = img_dir
        self.img_size = img_size
        self.max_missing_frames = max_missing_frames

        curr_location = 0
        missing_final_frame = 0
        possible_starts = []
        self.seq_counts = {}
        
        while curr_location < len(self.sources) - self.nt * self.frame_step + 1:
            
            count = self.seq_counts.get(self.sources[curr_location], 0)
            
            if self.max_seq_per_video is not None and count >= self.max_seq_per_video:
                curr_location += self.frame_step
            
            elif self.sources[curr_location] == self.sources[curr_location + self.nt * self.frame_step - 1]:
                
                #if self.sources[curr_location] == '33891':
                #    print(curr_location, curr_location + self.nt * self.frame_step - 1)
                
                possible_starts.append(curr_location)
                
                # keep sequence counts for debugging
                count = self.seq_counts.get(self.sources[curr_location], 0)
                self.seq_counts[self.sources[curr_location]] = count + 1
                
                old_location = curr_location
                curr_location += self.nt * self.frame_step
                curr_location -= self.seq_overlap * self.frame_step
                
            elif (curr_location + self.nt * self.frame_step - max_missing_frames - 1 > curr_location) and \
                 (self.sources[curr_location] == self.sources[curr_location + self.nt * self.frame_step - max_missing_frames - 1]):
                ''' If videos have varying number of frames, the last sequence may be incomplete.
                    In this case, we adjust the sequence start to guarantee the desired number of frames.
                    Depending on the task, it may be important to assure that all videos have the same 
                    number of sequences! '''
                
                last_frame_offset = None
                for i in range(1, max_missing_frames):
                    # look for last frame of current source video
                    
                    if self.sources[curr_location] == self.sources[curr_location + self.nt * self.frame_step - i - 1]:
                        last_frame_offset = i
                        break
                
                # If the last sequences has too few frames, we skip it.
                if last_frame_offset is None or curr_location - last_frame_offset < 0:
                    
                    #print('Last frame sequence of video "{}" is incomplete\
                    #(less than {} frames)'.format(self.sources[curr_location],                                                                           #                              self.nt * self.frame_step))
                    #print('Skipping this frame sequence...')
                    curr_location += self.frame_step
                
                # Otherwise, we ajust shift the sequence start.
                else:
                    curr_location -= last_frame_offset
                    missing_final_frame += 1
            else:
                curr_location += self.frame_step
                
        self.possible_starts = possible_starts
        print('Videos with missing frames: {}'.format(missing_final_frame))

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
            #index_array, current_index, current_batch_size = next(self.index_generator)
        
        #batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        current_batch_size = len(index_array)
        images = []
        
        # TODO: parallelize image loading
        for i, idx in enumerate(index_array):
            idx_start = self.possible_starts[idx]
            idx_end = idx_start + self.nt * self.frame_step
            
            # Avoid reading the same image multiple times
            if i < current_batch_size - 1:
                idx_start_next = self.possible_starts[idx+1]
                if idx_end > idx_start_next:
                    idx_end = idx_start_next
                
            images.extend(self.load_images(idx_start, idx_end))
            
        images = np.array(images)
        #print('images shape:', images.shape, images[0].shape)
        
        batch_x = []
        batch_y = []
        
        for i in range(current_batch_size):
            
            if self.seq_overlap > 0:
                start = i * self.seq_overlap
            else:
                start = i * self.nt
            
            if start < 0:
                start = 0
                
            end = start + self.nt
            
            #print('range:', start, end, len(images))
                
            if end <= len(images):
                batch_x.append(images[start:end])
                batch_y.append(self.sources[start:end])
                
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        if self.data_format == 'channels_first':
            batch_x = np.transpose(batch_x, (0, 1, 4, 2, 3))
            #images = np.transpose(images, (0, 3, 1, 2))
        
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
            
        return batch_x, batch_y
    
    def load_images(self, idx_start, idx_end):
        #images = np.zeros((self.nt,) + self.im_shape, np.float32)
        images = []
        
        for i in range(idx_start, idx_end, self.frame_step):
            category_dir = None
            frame_file = self.sources[i]
            
            if '__' in self.sources[i]:
                category_dir, frame_file = self.sources[i].split('__')
                
            frame_file = '{}__frame_{:03d}.jpg'.format(frame_file, (i-idx_start+1))
            #print('load img:', i, len(images))
            
            if category_dir:
                img = imread(os.path.join(self.img_dir, category_dir, frame_file))
            else:
                img = imread(os.path.join(self.img_dir, frame_file))
                
            images.append(self.preprocess(img))
        
        return images
            
    def crop_center(self, img, cropx, cropy):
        y,x,c = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[startx:startx+cropx,starty:starty+cropy]

    # resize and crop image
    def preprocess(self, im):
        larger_dim = np.argsort(self.img_size)[-1]
        smaller_dim = np.argsort(self.img_size)[-2]
        target_ds = float(self.img_size[larger_dim])/im.shape[larger_dim]

        im = resize(im, (int(np.round(target_ds * im.shape[0])), 
                         int(np.round(target_ds * im.shape[1]))),
                    mode='reflect')

        # crop
        im = self.crop_center(im, self.img_size[0], self.img_size[1])
        return im

