'''
Extract features from the pre-trained VGG.

Adapted from https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
'''
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical

from data import DataGenerator
from settings import configs
import argparse
import cPickle as pkl
from tqdm import tqdm


def get_create_results_dir(config_name, base_results_dir):
    results_dir = os.path.join(base_results_dir, config_name)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    return results_dir

def save_representation(features, labels, results_dir, config):
    
    for i, label in enumerate(labels):
        target_dir = results_dir
        if len(label) > 1:
            category, source = label
            target_dir = os.path.join(results_dir, category)
        else:
            source = label[0]
        
        if not os.path.exists(target_dir): os.makedirs(target_dir)
        
        features_file = '{}.pkl'.format(source)
        filename = os.path.join(target_dir, features_file)
        
        with open(filename, 'w') as f:
            pkl.dump(features[i], f)
            

def save_bottleneck_features(config_name, data_dir, base_results_dir, 
                             batch_size, input_shape, frames_per_video, 
                             max_videos_per_class, sample_step, 
                             classes=None, **config):
    
    max_frames_per_class = frames_per_video * max_videos_per_class
    
    generator = DataGenerator(batch_size=batch_size, return_sources=True,
                              fn_preprocess=preprocess_input,
                              shuffle=False, sample_step=sample_step, 
                              target_size=input_shape[:2], classes=classes,
                              max_per_class=max_frames_per_class)
    generator = generator.flow_from_directory(data_dir)
    output_generator = iter(generator)
    
    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet',
                  input_shape=generator.data_shape)

    results_dir = get_create_results_dir(config_name, base_results_dir)
    n_batches = len(generator)
    print('Number of batches: {}'.format(n_batches))
    
    for i in tqdm(range(n_batches)):
        X_, y_, sources = next(output_generator)
        features_train = model.predict(X_, generator.batch_size)
        y_batch = []
        
        for s in sources:
            path, source = os.path.split(s)
            path, category = os.path.split(path)
            path, data_split = os.path.split(path)
            #source = '__'.join(source.split('__')[:-1])
            source = source.replace('.jpg', '')
            category_source = (category, source)
            y_batch.append(category_source)
        
        target_dir = os.path.join(results_dir, data_split)
        save_representation(features_train, y_batch, target_dir, config)    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract VGG features.')
    parser.add_argument('config', help='experiment config name defined in settings.py')
    FLAGS, unparsed = parser.parse_known_args()
    
    config = configs[FLAGS.config]
    print('\n==> Starting feature extraction: {}'.format(config['description']))
    
    save_bottleneck_features(FLAGS.config, config['training_data_dir'], **config)
    save_bottleneck_features(FLAGS.config, config['validation_data_dir'], **config)