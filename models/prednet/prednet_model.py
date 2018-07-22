from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, TimeDistributed, Dense, Flatten
import numpy as np

from prednet import PredNet

def load_model(model_json_file, model_weights_file, **extras):   
    print('Loading model: {}'.format(model_weights_file))
    f = open(model_json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    train_model.load_weights(model_weights_file)
    return train_model

def create_model(model_json_file=None, model_weights_file=None, 
                 train=False, **config):
    if model_json_file and model_weights_file:
        pretrained_model = load_model(model_json_file, model_weights_file)
        model = pretrained_prednet(pretrained_model, train=train, **config)
    else:
        model = random_prednet(train=train, **config)
    return model

def pretrained_prednet(pretrained_model, output_mode='error', train=False, 
                       n_timesteps=10, stateful=False, batch_size=None, 
                       **config):
    layer_config = pretrained_model.layers[1].get_config()
    layer_config['output_mode'] = output_mode
    layer_config['stateful'] = stateful
    prednet = PredNet(weights=pretrained_model.layers[1].get_weights(), **layer_config)
    input_shape = list(pretrained_model.layers[0].batch_input_shape[1:])
    input_shape[0] = n_timesteps
    inputs = get_input_layer(batch_size, tuple(input_shape))
    outputs = get_output_layer(prednet, inputs, n_timesteps, train, output_mode)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def random_prednet(input_channels, input_height, input_width, 
                   stack_sizes=(48, 96, 192), n_timesteps=10, 
                   train=False, output_mode='error', stateful=False, 
                   batch_size=None, **config):
    # Model parameters
    if K.image_data_format() == 'channels_first':
        input_shape = (input_channels, input_height, input_width) 
    else:
        input_shape = (input_height, input_width, input_channels)
        
    stack_sizes = (input_channels,) + stack_sizes
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3,) * (len(stack_sizes) - 1)
    Ahat_filt_sizes = (3,) * len(stack_sizes)
    R_filt_sizes = (3,) * len(stack_sizes)
    prednet = PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode=output_mode, return_sequences=True, 
                      stateful=stateful)
    input_shape = (n_timesteps,) + input_shape
    inputs = get_input_layer(batch_size, input_shape)
    outputs = get_output_layer(prednet, inputs, n_timesteps, train, output_mode)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_input_layer(batch_size, input_shape):
    if batch_size:
        input_shape = (batch_size,) + input_shape
        inputs = Input(batch_shape=input_shape)
    else:
        inputs = Input(shape=input_shape)
    return inputs

def get_output_layer(prednet, inputs, n_timesteps, train, output_mode):
    outputs = prednet(inputs)
    if train:
        if output_mode != 'error':
            raise ValueError('When training, output_mode must be equal to "error"')
        outputs = get_error_layer(outputs, n_timesteps, prednet.nb_layers)
    return outputs

def get_error_layer(outputs, n_timesteps, nb_layers):
    # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.array([0.1] * nb_layers)
    layer_loss_weights[0] = 1
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
    # equally weight all timesteps except the first
    time_loss_weights = 1./ (n_timesteps - 1) * np.ones((n_timesteps,1))  
    time_loss_weights[0] = 0

    # calculate weighted error by layer
    errors_by_time = TimeDistributed(Dense(1, trainable=False), 
                                     weights=[layer_loss_weights, np.zeros(1)], 
                                     trainable=False)(outputs)
    errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, n_timesteps)
    final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], 
                         trainable=False)(errors_by_time) # weight errors by time
    return final_errors