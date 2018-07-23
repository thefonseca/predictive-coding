'''
Train PredNet on Moments in Time sequences.
Adapted from https://github.com/coxlab/prednet/blob/master/kitti_train.py
'''
import os
import numpy as np
import tensorflow as tf
import random as rn

from keras import backend as K
from keras.models import Model
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.callbacks import CSVLogger, EarlyStopping, LambdaCallback
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import Adam

# Getting reproducible results:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import utils
import prednet_model
import argparse
import sys
sys.path.append("../classifier")
from data import DataGenerator


class StateResetter(Callback):
    def on_batch_end(self, batch, logs={}):
        self.model.reset_states()
        
class CustomModelCheckpoint(ModelCheckpoint):
    '''
    Trick to use multi_gpu_model. We save the original model instead.
    See: https://github.com/keras-team/keras/issues/8649
    '''
    def __init__(self, model_to_save, *args, **kwargs):
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)
        self.model_to_save = model_to_save
        
    def on_epoch_end(self, epoch, logs=None):
        self.model = self.model_to_save
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)

def get_callbacks(model, results_dir, stopping_patience=None, stateful=False):
    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    
    callbacks = [LearningRateScheduler(lr_schedule)]
    
    checkpoint_path = os.path.join(results_dir, 'weights.hdf5')
    csv_path = os.path.join(results_dir, 'train.log')
    checkpointer = CustomModelCheckpoint(filepath=checkpoint_path,
                                         monitor='val_loss',
                                         model_to_save=model,
                                         verbose=1, save_best_only=True)
    
    csv_logger = CSVLogger(csv_path)
    callbacks.append(checkpointer)
    callbacks.append(csv_logger)
    if stopping_patience:
        stopper = EarlyStopping(monitor='val_loss', 
                                patience=stopping_patience, 
                                verbose=0, mode='auto')
        callbacks.append(stopper)
    
    if stateful:
        callbacks.append(StateResetter())
    return callbacks
    
def train(config_name, training_data_dir, validation_data_dir, 
          base_results_dir, test_data_dir=None, epochs=150, 
          use_multiprocessing=False, workers=1, shuffle=True,
          n_timesteps=10, batch_size=4, stopping_patience=None, 
          input_channels=3, input_width=160, input_height=128, 
          max_queue_size=10, classes=None, 
          training_index_start=0, training_max_per_class=None, 
          frame_step=1, stateful=False, rescale=None, gpus=None,
          validation_index_start=0, validation_max_per_class=None, 
          data_format=K.image_data_format(), 
          seq_overlap=0, **config):
    
    model = prednet_model.create_model(train=True, stateful=stateful,
                                       input_channels=input_channels, 
                                       input_width=input_width, 
                                       batch_size=batch_size,
                                       input_height=input_height, **config)
    
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    callbacks = get_callbacks(model, results_dir, stopping_patience, stateful)
    
    json_file = os.path.join(results_dir, 'model.json')
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
    
    if gpus:
        model = multi_gpu_model(model, gpus=gpus)
    
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer='adam')
    
    layer_config = model.layers[1].get_config()
    data_format = layer_config['data_format'] if 'data_format' in layer_config else data_format #layer_config['dim_ordering']
    
    resize = lambda img: utils.resize_img(img, target_size=(input_height, 
                                                            input_width))
    
    train_generator = DataGenerator(classes=classes,
                                    seq_length=n_timesteps,
                                    min_seq_length=n_timesteps,
                                    seq_overlap=seq_overlap,
                                    sample_step=frame_step,
                                    target_size=None,
                                    rescale=rescale,
                                    fn_preprocess=resize,
                                    batch_size=batch_size, 
                                    shuffle=shuffle,
                                    data_format=data_format,
                                    output_mode='error',
                                    index_start=training_index_start,
                                    max_per_class=training_max_per_class)
    
    val_generator = DataGenerator(classes=classes,
                                  seq_length=n_timesteps,
                                  min_seq_length=n_timesteps,
                                  seq_overlap=seq_overlap,
                                  sample_step=frame_step,
                                  target_size=None,
                                  rescale=rescale,
                                  fn_preprocess=resize,
                                  batch_size=batch_size,
                                  data_format=data_format,
                                  output_mode='error', 
                                  index_start=validation_index_start,
                                  max_per_class=validation_max_per_class)
    
    train_generator = train_generator.flow_from_directory(training_data_dir)
    val_generator = val_generator.flow_from_directory(validation_data_dir)
    
    if len(train_generator) == 0 or len(val_generator) == 0:
        return
    
    history = model.fit_generator(train_generator, 
                                  len(train_generator), 
                                  epochs, callbacks=callbacks, 
                                  validation_data=val_generator, 
                                  validation_steps=len(val_generator), 
                                  use_multiprocessing=use_multiprocessing,
                                  max_queue_size=max_queue_size, 
                                  workers=workers)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PredNet model.')
    parser.add_argument('config', help='experiment config name defined in moments_settings.py')
    parser.add_argument('--stateful', help='use stateful PredNet model', action='store_true')
    parser.add_argument('--task', help='use stateful PredNet model', choices=['3c', '10c', 'full'])
    parser.add_argument('--gpus', type=int, help='number of gpus')
    FLAGS, unparsed = parser.parse_known_args()
    
    config_name, config = utils.get_config(vars(FLAGS))
    
    print('\n==> Starting experiment: {}\n'.format(config['description']))
    config_str = utils.get_config_str(config)
    print('==> Using configuration:\n{}'.format(config_str))
    
    train(config_name, **config)
    utils.save_experiment_config(config_name, config['base_results_dir'], config)