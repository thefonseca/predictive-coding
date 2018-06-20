from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger

from settings import configs
from convnet_data import DataGenerator
import os
import argparse


def get_model(input_shape):
    
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Conv2D(64, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    return model
    
def train(config_name, data_dir, base_results_dir, epochs=10, batch_size=10, **config):
    
    training_data_dir = os.path.join(data_dir, 'training')
    validation_data_dir = os.path.join(data_dir, 'validation')
    training_generator = DataGenerator(training_data_dir, batch_size=batch_size)
    validation_generator = DataGenerator(validation_data_dir, batch_size=batch_size)
    
    model = get_model(training_generator.data_shape)
    
    results_dir = os.path.join(base_results_dir, config_name)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    checkpoint_path = os.path.join(results_dir, 'weights.hdf5')
        
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, 
                                   verbose=1, save_best_only=True)
    csv_path = os.path.join(results_dir, 'training.log')
    csv_logger = CSVLogger(csv_path)
    
    model.fit_generator(generator=training_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        callbacks=[checkpointer, csv_logger],
                        use_multiprocessing=False, workers=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train convnet classifier.')
    parser.add_argument('config', help='experiment config name defined in setting.py')
    FLAGS, unparsed = parser.parse_known_args()
    
    config = configs[FLAGS.config]
    print('\n==> Starting traininig: {}'.format(config['description']))
    
    train(FLAGS.config, **config)

