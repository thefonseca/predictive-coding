import os
import argparse
import numpy as np
from tensorflow import set_random_seed
np.random.seed(17)
set_random_seed(17)

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.models import load_model

from data import DataGenerator
from settings import configs
import models
import utils

def save_experiment_config(config_name, base_results_dir, config):
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    f = open(os.path.join(results_dir, 'experiment_config.txt'), 'w')
    for key in sorted(config):
        f.write('{}: {}\n'.format(key, config[key]))
    f.close()

    
def train(config_name, training_data_dir, validation_data_dir, 
          base_results_dir, test_data_dir=None, epochs=10, 
          use_multiprocessing=False, workers=1, dropout=0.5, 
          seq_length=None, batch_size=10, stopping_patience=0, 
          classes=None, input_shape=None, max_queue_size=10, 
          model_type='convnet', **config):
    
    max_per_class = config.get('training_max_per_class', None)
    train_generator = DataGenerator(batch_size=batch_size,
                                    classes=classes,
                                    seq_length=seq_length,
                                    target_size=input_shape,
                                    max_per_class=max_per_class)
    
    val_generator = DataGenerator(classes=classes,
                                  seq_length=seq_length,
                                  target_size=input_shape,
                                  batch_size=batch_size)
    
    train_generator = train_generator.flow_from_directory(training_data_dir)
    val_generator = val_generator.flow_from_directory(validation_data_dir)
    
    input_shape = train_generator.data_shape
    n_classes = train_generator.n_classes
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    
    '''models = { 'convnet': convnet, 
               'convlstm': convlstm,
               'lstm': lstm }'''
    model = getattr(models, model_type)(input_shape, n_classes, drop_rate=dropout)
    #model = models[model_type](input_shape, n_classes, drop_rate=dropout)
    checkpoint_path = os.path.join(results_dir, model_type + '.hdf5')
    csv_path = os.path.join(results_dir, model_type + '.log')

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
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
             seq_length=None, input_shape=None, **config):
    
    print('\nEvaluating model on test set...')
    # we use the remaining part of training set as test set
    max_per_class = config.get('test_max_per_class', None)
    generator = DataGenerator(classes=classes,
                              batch_size=batch_size,
                              seq_length=seq_length,
                              index_start=index_start, 
                              target_size=input_shape,
                              max_per_class=max_per_class)
    generator = generator.flow_from_directory(test_data_dir)
    
    # load best model
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    
    if seq_length:
        checkpoint_path = os.path.join(results_dir, 'convlstm.hdf5')
    else:
        checkpoint_path = os.path.join(results_dir, 'convnet.hdf5')    
    
    model = load_model(checkpoint_path)
    metrics = model.evaluate_generator(generator,
                                       len(generator),
                                       use_multiprocessing=use_multiprocessing, 
                                       workers=workers)

    metric_str = ['{}: {}'.format(m, v) for m, v in zip(model.metrics_names, metrics)]
    metric_str = ' - '.join(metric_str)
    print('Test {}'.format(metric_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train convnet classifier.')
    parser.add_argument('config', help='experiment config name defined in setting.py')
    FLAGS, unparsed = parser.parse_known_args()
    
    config = configs[FLAGS.config]
    print('\n==> Starting traininig: {}'.format(config['description']))
    
    train(FLAGS.config, **config)
    
    if config['test_data_dir']:
        evaluate(FLAGS.config, index_start=config['training_max_per_class'], **config)

    save_experiment_config(FLAGS.config, config['base_results_dir'], config)