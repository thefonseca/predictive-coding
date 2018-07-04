from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, TimeDistributed, Dense, Flatten
import os
import numpy as np
from skimage.transform import resize

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
                 train=False, **config):
    if model_json_file and model_weights_file:
        pretrained_model = load_model(model_json_file, model_weights_file)
        model = pretrained_prednet(pretrained_model, train=train, **config)
    else:
        model = random_prednet(train=train, **config)
    
    #if gpus:
    #    model = multi_gpu_model(model, gpus=gpus)
    return model

def pretrained_prednet(pretrained, output_mode='error', train=False, 
                       n_timesteps=10, stateful=False, batch_size=None, 
                       **config):
    layer_config = pretrained.layers[1].get_config()
    layer_config['output_mode'] = output_mode
    layer_config['stateful'] = stateful
    prednet = PredNet(weights=pretrained.layers[1].get_weights(), **layer_config)
    input_shape = list(pretrained.layers[0].batch_input_shape[1:])
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
    print(stack_sizes)
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
    
def get_config_str(config):
    config_str = ''
    for k, v in sorted(config.items()):
        if k != 'description':
            config_str += '    {}: {}\n'.format(k, v)
    return config_str

def crop_center(img, cropx, cropy):
        y,x,c = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[startx:startx+cropx,starty:starty+cropy]

def resize_img(img, target_size):
    larger_dim = np.argsort(target_size)[-1]
    smaller_dim = np.argsort(target_size)[-2]
    target_ds = float(target_size[larger_dim])/img.shape[larger_dim]

    img = resize(img, (int(np.round(target_ds * img.shape[0])), 
                       int(np.round(target_ds * img.shape[1]))),
                 mode='reflect')
    
    # crop
    img = crop_center(img, target_size[0], target_size[1])
    return img

def get_config(configs, FLAGS):
    config = configs[FLAGS.config]
    
    if not FLAGS.task:
        suffix = '__' + config['task']
    else:
        suffix = '__' + FLAGS.task
        config['task'] = FLAGS.task
    
    if FLAGS.stateful is None:
        stateful = config['stateful']
    else:
        stateful = FLAGS.stateful
        config['stateful'] = FLAGS.stateful
    
    if stateful:
        suffix += '__' + 'stateful'
    
    if config['model_weights_file']:
        config['model_weights_file'] = config['model_weights_file'].format(suffix)
    if config['model_json_file']:
        config['model_json_file'] = config['model_json_file'].format(suffix)
        
    name = FLAGS.config + suffix
    config['config_name'] = name
    return name, config