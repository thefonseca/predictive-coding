from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.models import load_model

from settings import configs
from convlstm_data import DataGenerator
import os
import argparse
import utils


def convnet(input_shape, n_classes, drop_rate=0.5):
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
              input_shape=input_shape, 
              activation='relu'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    #model.add(Conv2D(32, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    return model

def convlstm(input_shape, n_classes, drop_rate=0.5):
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=10, kernel_size=(3, 3),
                       input_shape=input_shape,
                       padding='same', return_sequences=True))
    #model.add(BatchNormalization())

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
    return model


def lstm(input_shape, n_classes, drop_rate=0.5):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(Flatten(input_shape=self.input_shape))
    model.add(LSTM(32, return_sequences=False, dropout=drop_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(n_classes, activation='softmax'))


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
          classes=None, **config):
    
    max_per_class = config.get('training_max_per_class', None)
    train_generator = DataGenerator(batch_size=batch_size,
                                    classes=classes,
                                    seq_length=seq_length,
                                    max_per_class=max_per_class)
    
    val_generator = DataGenerator(classes=classes,
                                  seq_length=seq_length,
                                  batch_size=batch_size)
    
    train_generator = train_generator.flow_from_directory(training_data_dir)
    val_generator = val_generator.flow_from_directory(validation_data_dir)
    
    input_shape = train_generator.data_shape
    n_classes = train_generator.n_classes
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    
    if seq_length:
        model = convlstm(input_shape, n_classes, drop_rate=dropout)
        checkpoint_path = os.path.join(results_dir, 'convlstm.hdf5')
        csv_path = os.path.join(results_dir, 'convnet.log')
    else:
        model = convnet(input_shape, n_classes, drop_rate=dropout)
        checkpoint_path = os.path.join(results_dir, 'convnet.hdf5')    
        csv_path = os.path.join(results_dir, 'convnet.log')
        
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
                        workers=workers)
    

def evaluate(config_name, test_data_dir, batch_size, 
             index_start, base_results_dir, classes=None,
             workers=1, use_multiprocessing=False, 
             seq_length=None, **config):
    
    print('\nEvaluating model on test set...')
    # we use the remaining part of training set as test set
    generator = DataGenerator(classes=classes,
                              batch_size=batch_size,
                              seq_length=seq_length,
                              index_start=index_start)
    generator = generator.flow_from_directory(test_data_dir)
    
    # load best model
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
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