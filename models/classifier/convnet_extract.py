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

from conv_lstm_data import DataGenerator
from settings import configs
import argparse


def get_create_results_dir(config_name, base_results_dir):
    results_dir = os.path.join(base_results_dir, config_name)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    return results_dir

def save_bottleneck_features(config_name, base_results_dir, 
                             training_data_dir, validation_data_dir, 
                             batch_size, input_shape, frames_per_video, 
                             max_videos_per_class, sample_step, **config):
    
    max_frames_per_class = frames_per_video * max_videos_per_class
    training_generator = DataGenerator(batch_size=batch_size, 
                                       fn_preprocess=preprocess_input,
                                       shuffle=False, sample_step=sample_step, 
                                       target_size=input_shape[:2])
    training_generator = training_generator.flow_from_directory(training_data_dir)
    
    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet',
                  input_shape=training_generator.data_shape)

    features_train = model.predict_generator(training_generator, 
                                             len(training_generator),
                                             verbose=1)
    
    print(features_train.shape)
    
    features_train = {
        'X': features_train,
        'y': training_generator.y[:features_train.shape[0]],
        'sources': training_generator.X[:features_train.shape[0]]
    }
    
    results_dir = get_create_results_dir(config_name, base_results_dir)
    features_file = os.path.join(results_dir, 'bottleneck_features_train.npy')
    np.save(open(features_file, 'w'), features_train)

    validation_generator = DataGenerator(batch_size=batch_size,
                                         fn_preprocess=preprocess_input,
                                         shuffle=False, sample_step=sample_step,
                                         target_size=input_shape[:2])
    
    validation_generator = validation_generator.flow_from_directory(validation_data_dir)
    
    features_validation = model.predict_generator(validation_generator, 
                                                  len(validation_generator), 
                                                  verbose=1)
    
    print(features_validation.shape)
    
    features_validation = {
        'X': features_validation,
        'y': validation_generator.y[:features_validation.shape[0]],
        'sources': validation_generator.X[:features_validation.shape[0]]
    }
    
    features_file = os.path.join(results_dir, 'bottleneck_features_validation.npy')
    np.save(open(features_file, 'w'), features_validation)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract VGG features.')
    parser.add_argument('config', help='experiment config name defined in setting.py')
    FLAGS, unparsed = parser.parse_known_args()
    
    config = configs[FLAGS.config]
    print('\n==> Starting feature extraction: {}'.format(config['description']))
    
    save_bottleneck_features(FLAGS.config, **config)