from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D, LSTM, TimeDistributed
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K


def convnet(input_shape, n_classes, drop_rate=0.5):
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
              input_shape=input_shape, 
              activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))
    model.add(Flatten())
    model.add(Dense(512))
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
    model.add(ConvLSTM2D(filters=200, kernel_size=(3, 3),
                       input_shape=input_shape, dropout=drop_rate,
                       padding='same', return_sequences=True))
    #model.add(BatchNormalization())
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
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])
    
    model = Sequential()
    model.add(TimeDistributed(Flatten(), input_shape=input_shape))
    model.add(LSTM(512, return_sequences=False, dropout=drop_rate))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    return model