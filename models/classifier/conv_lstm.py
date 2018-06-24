from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model

from settings import configs
from conv_lstm_data import DataGenerator
import os
import argparse


def get_model(input_shape, n_classes, drop_rate=0.5):
    
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=input_shape,
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())

    #seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
    #                   padding='same', return_sequences=True))
    #seq.add(BatchNormalization())

    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))
    
    model.add(Flatten())
    model.add(Dense(32))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    return model

def get_create_results_dir(config_name, base_results_dir):
    results_dir = os.path.join(base_results_dir, config_name)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    return results_dir
    
def train(config_name, training_data_dir, validation_data_dir, 
          base_results_dir, test_data_dir=None, epochs=10, workers=1,
          use_multiprocessing=False, batch_size=10, seq_length=20, 
          dropout=0.5, **config):
    
    max_per_class = config.get('training_max_per_class', None)
    training_generator = DataGenerator(batch_size=batch_size,
                                       max_per_class=max_per_class, 
                                       seq_length=seq_length)
    
    validation_generator = DataGenerator(batch_size=batch_size, 
                                         seq_length=seq_length)
    
    training_generator = training_generator.flow_from_directory(training_data_dir)
    validation_generator = validation_generator.flow_from_directory(validation_data_dir)
    
    print(len(training_generator), len(validation_generator))
    
    model = get_model(training_generator.data_shape, 
                      training_generator.n_classes,
                      drop_rate=dropout)
    
    results_dir = get_create_results_dir(config_name, base_results_dir)
    checkpoint_path = os.path.join(results_dir, 'conv_lstm.hdf5')
        
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, 
                                   verbose=1, save_best_only=True)
    csv_path = os.path.join(results_dir, 'conv_lstm.log')
    csv_logger = CSVLogger(csv_path)
    
    model.fit_generator(training_generator,
                        len(training_generator),
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator),
                        callbacks=[checkpointer, csv_logger],
                        use_multiprocessing=use_multiprocessing, 
                        workers=workers)
    
def evaluate(config_name, test_data_dir, batch_size, 
             index_start, base_results_dir, workers=1,
             use_multiprocessing=False, seq_length=20, **config):
    
    print('\nEvaluating model on test set...')
    # we use the remaining part of training set as test set
    test_generator = DataGenerator(batch_size=batch_size, 
                                   seq_length=seq_length,
                                   index_start=index_start)
    test_generator = test_generator.flow_from_directory(test_data_dir)

    # load best model
    results_dir = get_create_results_dir(config_name, base_results_dir)
    checkpoint_path = os.path.join(results_dir, 'conv_lstm.hdf5')
    model = load_model(checkpoint_path)
    
    metrics = model.evaluate_generator(test_generator,
                                       len(test_generator),
                                       use_multiprocessing=use_multiprocessing, 
                                       workers=workers)

    metric_str = ['{}: {}'.format(m, v) for m, v in zip(model.metrics_names, metrics)]
    metric_str = ' - '.join(metric_str)
    print('Test {}'.format(metric_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ConvLSTM classifier.')
    parser.add_argument('config', help='experiment config name defined in setting.py')
    FLAGS, unparsed = parser.parse_known_args()
    
    config = configs[FLAGS.config]
    print('\n==> Starting traininig: {}'.format(config['description']))
    
    train(FLAGS.config, **config)
    
    if config['test_data_dir']:
        evaluate(FLAGS.config, index_start=config['training_max_per_class'], **config)

