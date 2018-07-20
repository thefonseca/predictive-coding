from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D, LSTM, TimeDistributed
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Input, Average, Masking, Reshape
from keras import backend as K



def convnet(input_shape, n_classes, drop_rate=0.5):
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
    
    model = Sequential()
    model.add(Conv2D(100, (3, 3), padding='same',
              input_shape=input_shape, 
              activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))
    model.add(Flatten())
    model.add(Dense(100))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    return model

def convlstm(input_shape, n_classes, drop_rate=0.5, mask_value=None):
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])
    
    inputs = Input(shape=input_shape)
    if mask_value is not None:
        x = TimeDistributed(Flatten())(inputs)
        x = TimeDistributed(Masking(mask_value=mask_value))(x)
        x = TimeDistributed(Reshape(input_shape[1:]))(x)
    else:
        x = inputs
    x = ConvLSTM2D(filters=200, kernel_size=(3, 3), dropout=drop_rate,
                   padding='same', return_sequences=True)(x)
    x = Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid',
               padding='same', data_format='channels_last')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(drop_rate)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=[predictions])

def lstm(input_shape, n_classes, drop_rate=0.5, mask_value=None):
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])
    
    inputs = Input(shape=input_shape)
    x = TimeDistributed(Flatten())(inputs)
    if mask_value is not None:
        x = Masking(mask_value=mask_value)(x)
    x = LSTM(64, return_sequences=False, dropout=drop_rate)(x)
    #x = Dense(32, activation='relu')(x)
    #x = Dropout(drop_rate)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=predictions)
    
def ensemble(models, input_shape):
    if models and len(models) < 2:
        raise ValueError('To get an ensemble you need at least two models')

    for i, model in enumerate(models):
        for l in model.layers:
            l.name = l.name + '_ens_{}'.format(i)
            
    inputs = [inp for model in models for inp in model.inputs]
    outputs = [inp for model in models for inp in model.outputs]
    avg = Average()(outputs)
    ensemble = Model(inputs=inputs, outputs=avg)
    return ensemble