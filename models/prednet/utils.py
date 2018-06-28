from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, TimeDistributed, Dense, Flatten
import os
import numpy as np

from prednet import PredNet

def load_model(model_json_file, model_weights_file, **extras):
    # Load trained model
    f = open(model_json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    train_model.load_weights(model_weights_file)
    return train_model

def create_model(model_json_file=None, model_weights_file=None, 
                 train=False, n_timesteps=10, **config):
    if model_json_file and model_weights_file:
        pretrained_model = load_model(model_json_file, model_weights_file)
        model = pretrained_prednet(pretrained_model, train=train, **config)
    else:
        model = prednet(train=train, **config)
        
    #if gpus:
    #    model = multi_gpu_model(model, gpus=gpus)
    return model

def pretrained_prednet(pretrained, output_mode='error', 
                       train=False, n_timesteps=10, **config):
    # Create testing model (to output predictions)
    layer_config = pretrained.layers[1].get_config()
    layer_config['output_mode'] = output_mode
    prednet = PredNet(weights=pretrained.layers[1].get_weights(), **layer_config)
    input_shape = list(pretrained.layers[0].batch_input_shape[1:])
    input_shape[0] = n_timesteps
    inputs = Input(shape=tuple(input_shape))
    outputs = prednet(inputs)
    
    if train:
        outputs = get_error_layer(outputs, n_timesteps)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def prednet(n_channels, img_height, img_width, n_timesteps=10, 
            train=False, output_mode='error', **config):
    # Model parameters
    if K.image_data_format() == 'channels_first':
        input_shape = (n_channels, img_height, img_width) 
    else:
        input_shape = (img_height, img_width, n_channels)
        
    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    
    prednet = PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode=output_mode, return_sequences=True) 
    input_shape = (n_timesteps,) + input_shape
    inputs = Input(shape=input_shape)
    outputs = prednet(inputs)
    
    if train:
        outputs = get_error_layer(outputs, n_timesteps)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_error_layer(outputs, n_timesteps):
    # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.array([1, 0.1, 0.1, 0.1])  
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

def get_create_results_dir(experiment_name, base_results_dir, dataset=None):
    results_dir = os.path.join(base_results_dir, experiment_name)
    if dataset:
        results_dir = os.path.join(results_dir, dataset)
    
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    return results_dir

def save_experiment_config(experiment_name, base_results_dir, config, dataset=None):   
    results_dir = get_create_results_dir(experiment_name, 
                                         base_results_dir, 
                                         dataset=dataset)
    f = open(os.path.join(results_dir, 'experiment_config.txt'), 'w')
    for key in sorted(config):
        f.write('{}: {}\n'.format(key, config[key]))
    f.close()