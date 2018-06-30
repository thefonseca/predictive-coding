import os
import argparse

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.models import load_model

from data import DataGenerator
from settings import configs
import models
import utils

import numpy as np
import tensorflow as tf
import random as rn

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

def save_experiment_config(config_name, base_results_dir, config):
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    f = open(os.path.join(results_dir, 'experiment_config.txt'), 'w')
    for key in sorted(config):
        f.write('{}: {}\n'.format(key, config[key]))
    f.close()

    
def train(config_name, training_data_dir, validation_data_dir, 
          base_results_dir, test_data_dir=None, epochs=10, 
          use_multiprocessing=False, workers=1, dropout=0.5, 
          seq_length=None, sample_step=1, batch_size=10, 
          stopping_patience=3, classes=None, input_shape=None, 
          max_queue_size=10, model_type='convnet', shuffle=False,
          training_max_per_class=None, **config):
    
    train_generator = DataGenerator(batch_size=batch_size,
                                    shuffle=shuffle,
                                    classes=classes,
                                    seq_length=seq_length,
                                    target_size=input_shape,
                                    sample_step=sample_step,
                                    max_per_class=training_max_per_class)
    
    val_generator = DataGenerator(batch_size=batch_size,
                                  shuffle=shuffle,
                                  classes=classes,
                                  seq_length=seq_length,
                                  target_size=input_shape,
                                  sample_step=sample_step)
    
    train_generator = train_generator.flow_from_directory(training_data_dir)
    val_generator = val_generator.flow_from_directory(validation_data_dir)
    
    input_shape = train_generator.data_shape
    n_classes = train_generator.n_classes
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    
    model = getattr(models, model_type)(input_shape, n_classes, drop_rate=dropout)
    checkpoint_path = os.path.join(results_dir, model_type + '.hdf5')
    csv_path = os.path.join(results_dir, model_type + '.log')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, 
                                   verbose=1, save_best_only=True)
    
    csv_logger = CSVLogger(csv_path)
    stopper = EarlyStopping(monitor='val_loss', 
                            patience=stopping_patience, 
                            verbose=0, mode='auto')
    
    model.fit_generator(train_generator,
                        len(train_generator),
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        callbacks=[checkpointer, csv_logger, stopper],
                        use_multiprocessing=use_multiprocessing,
                        max_queue_size=max_queue_size, 
                        workers=workers)
    

def evaluate(config_name, test_data_dir, batch_size, 
             index_start, base_results_dir, classes=None,
             workers=1, use_multiprocessing=False,
             seq_length=None, input_shape=None, 
             model_type='convnet', test_max_per_class=None,
             **config):
    
    print('\nEvaluating model on test set...')
    # we use the remaining part of training set as test set
    generator = DataGenerator(classes=classes,
                              batch_size=batch_size,
                              seq_length=seq_length,
                              index_start=index_start, 
                              target_size=input_shape,
                              max_per_class=test_max_per_class)
    generator = generator.flow_from_directory(test_data_dir)
    
    # load best model
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    checkpoint_path = os.path.join(results_dir, model_type + '.hdf5')       
    model = load_model(checkpoint_path)
    metrics = model.evaluate_generator(generator, len(generator),
                                       use_multiprocessing=use_multiprocessing, 
                                       workers=workers)

    metric_str = ['{}: {}'.format(m, v) for m, v in zip(model.metrics_names, metrics)]
    metric_str = ' - '.join(metric_str)
    print('Test {}'.format(metric_str))
    f = open(os.path.join(results_dir, 'test.txt'), 'w')
    f.write('Test results:\n')
    f.write(metric_str)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument('config', help='experiment config name defined in setting.py')
    parser.add_argument('--eval', help="perform only evaluation using pretrained model",
                        action="store_true")
    FLAGS, unparsed = parser.parse_known_args()
    
    config = configs[FLAGS.config]
    print('\n==> Starting experiment: {}'.format(config['description']))
    config_str = utils.get_config_str(config)
    print('\n==> Using configuration:\n{}'.format(config_str))
    
    if not FLAGS.eval:
        train(FLAGS.config, **config)
        save_experiment_config(FLAGS.config, config['base_results_dir'], config)
    
    if config['test_data_dir']:
        evaluate(FLAGS.config, index_start=config['training_max_per_class'], **config)

    