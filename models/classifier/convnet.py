from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model

from settings import configs
from convlstm_data import DataGenerator
import os
import argparse
import utils


def get_model(input_shape, n_classes, drop_rate=0.25):
    
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
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    return model
    
def train(config_name, training_data_dir, validation_data_dir, 
          base_results_dir, test_data_dir=None, epochs=10, 
          use_multiprocessing=False, workers=1,
          batch_size=10, classes=None, **config):
    
    max_per_class = config.get('training_max_per_class', None)
    training_generator = DataGenerator(batch_size=batch_size,
                                       classes=classes,
                                       max_per_class=max_per_class)
    
    validation_generator = DataGenerator(classes=classes,
                                         batch_size=batch_size)
    
    training_generator = training_generator.flow_from_directory(training_data_dir)
    validation_generator = validation_generator.flow_from_directory(validation_data_dir)
    
    model = get_model(training_generator.data_shape, 
                      training_generator.n_classes)
    
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    checkpoint_path = os.path.join(results_dir, 'convnet.hdf5')
        
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, 
                                   verbose=1, save_best_only=True)
    csv_path = os.path.join(results_dir, 'convnet.log')
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
             index_start, base_results_dir, classes=None,
             workers=1, use_multiprocessing=False, 
             seq_length=20, **config):
    
    print('\nEvaluating model on test set...')
    # we use the remaining part of training set as test set
    generator = DataGenerator(classes=classes,
                              batch_size=batch_size, 
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
    
    #train(FLAGS.config, **config)
    
    if config['test_data_dir']:
        evaluate(FLAGS.config, index_start=config['training_max_per_class'], **config)

