from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D, LSTM, TimeDistributed
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Input, Average, Masking, Reshape, Lambda
from keras.layers import Bidirectional, Concatenate
from keras import backend as K

import sys
sys.path.append("../prednet")
import prednet_model


def convnet(input_shape, n_classes, hidden_dims, drop_rate=0.5):
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
    for dim in hidden_dims:
        model.add(Dense(dim, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    return model

def convlstm(input_shape, n_classes, hidden_dims, drop_rate=0.5, mask_value=None):
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
    for dim in hidden_dims:
        x = Dense(dim, activation='relu')(x)
        x = Dropout(drop_rate)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=[predictions])

def lstm(input_shape, n_classes, hidden_dims, 
         drop_rate=0.5, mask_value=None, **config):
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])
    
    inputs = Input(shape=input_shape)
    x = TimeDistributed(Flatten())(inputs)
    if mask_value is not None:
        x = Masking(mask_value=mask_value)(x)
    for dim in hidden_dims:
        x = LSTM(dim, return_sequences=False, dropout=drop_rate)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=predictions)

def crop(dimension, start, end=None):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    # See https://github.com/keras-team/keras/issues/890
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def lstm_layer(tensor, mask_value, hidden_dims, dropout):
    x = TimeDistributed(Flatten())(tensor)
    if mask_value is not None:
        x = Masking(mask_value=mask_value)(x)
    for dim in hidden_dims:
        x = Bidirectional(LSTM(dim, return_sequences=False, dropout=dropout), 
                          merge_mode='concat')(x)
    return x

def prednet_lstm(input_shape, n_classes, hidden_dims, 
                 drop_rate=0.5, mask_value=None, **config):
    if config is None:
        config = {}
    config['input_width'] = input_shape[0]
    config['input_height'] = input_shape[1]
    config['input_channels'] = input_shape[2]
        
    model = prednet_model.create_model(train=False, 
                                       output_mode='representation', 
                                       **config)
    prednet_layer = model.layers[1]
    for l in model.layers:
        l.trainable = False
    
    image_input = model.inputs[0]
    image = lstm_layer(image_input, mask_value, hidden_dims, drop_rate)
    
    index = 0
    reps = []
    flat_shapes = [61440, 245760, 122880, 61440]
    for l in range(prednet_layer.nb_layers):
        if l not in [1, 2]:
            reps.append(crop(2, index, index + flat_shapes[l])(model.outputs[0]))
        index += flat_shapes[l]
        
    lstms = []
    for r in reps:
        lstms.append(lstm_layer(r, mask_value, hidden_dims, drop_rate))
    
    x = Concatenate(axis=1)([image] + [l for l in lstms])
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=model.inputs, outputs=predictions)
    return model

def prednet(input_shape, n_classes, hidden_dims, 
            drop_rate=0.5, mask_value=None, **config):
    if config is None:
        config = {}
    
    config['input_height'] = input_shape[1]
    config['input_width'] = input_shape[2]
    config['input_channels'] = input_shape[3]
        
    model = prednet_model.create_model(train=False, 
                                       output_mode='representation', 
                                       **config)
    x = crop(1, -1)(model.outputs[0])   
    x = Flatten()(x)
    
    for dim in hidden_dims:
        x = Dense(dim, activation='relu')(x)
        x = Dropout(drop_rate)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=model.inputs, outputs=predictions)
    return model
    
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