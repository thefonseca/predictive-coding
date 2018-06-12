'''
Evaluate trained PredNet on Moments in Time sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from keras.utils import multi_gpu_model

from prednet import PredNet
from moments_data import SequenceGenerator
from moments_settings import experiments

from tqdm import tqdm
import argparse

FLAGS = None


def load_model(model_json_file, model_weights_file, **extras):
    # Load trained model
    f = open(model_json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    train_model.load_weights(model_weights_file)
    return train_model

def create_test_model(train_model, output_mode, n_timesteps=10, gpus=None, **extras):
    # Create testing model (to output predictions)
    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = output_mode
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = n_timesteps
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    test_model = Model(inputs=inputs, outputs=predictions)
    
    if gpus:
        test_model = multi_gpu_model(test_model, gpus=gpus)
        
    return test_model

def save_predictions(X, X_hat, experiment_name, results_dir, output_mode,
                     n_plot=20, **extras):
    
    # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
    mse_model = np.mean((X[:, 1:] - X_hat[:, 1:]) ** 2)  # look at all timesteps except the first
    mse_prev = np.mean((X[:, :-1] - X[:, 1:]) ** 2)
    
    results_dir = os.path.join(results_dir, experiment_name + '_' + output_mode)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    f = open(os.path.join(results_dir, 'prediction_scores.txt'), 'w')
    f.write("Model MSE: %f\n" % mse_model)
    f.write("Previous Frame MSE: %f" % mse_prev)
    f.close()

    # Plot some predictions
    n_timesteps =  X.shape[1]
    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
    plt.figure(figsize = (n_timesteps, 2 * aspect_ratio))
    gs = gridspec.GridSpec(2, n_timesteps)
    gs.update(wspace=0., hspace=0.)
    plot_save_dir = os.path.join(results_dir, 'prediction_plots/')
    if not os.path.exists(plot_save_dir): os.makedirs(plot_save_dir)
    
    plot_idx = np.random.permutation(X.shape[0])[:n_plot]
    
    for i in plot_idx:
        for t in range(n_timesteps):
            plt.subplot(gs[t])
            plt.imshow(X[i,t], interpolation='none')
            plt.tick_params(axis='both', which='both', 
                            bottom=False, top=False, 
                            left=False, right=False, 
                            labelbottom=False, labelleft=False)
            if t==0: plt.ylabel('Actual', fontsize=10)

            plt.subplot(gs[t + n_timesteps])
            plt.imshow(X_hat[i,t], interpolation='none')
            plt.tick_params(axis='both', which='both', 
                            bottom=False, top=False, 
                            left=False, right=False, 
                            labelbottom=False, labelleft=False)
            if t==0: plt.ylabel('Predicted', fontsize=10)

        plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
        plt.clf()

def evaluate(img_dir, img_sources, output_mode, n_timesteps=10, 
             frame_step=3, seq_overlap=5, max_seq_per_video=5, 
             shuffle=False, batch_size=5, max_missing_frames=15, 
             N_seq=None, seed=17, data_format=K.image_data_format(), 
             **extras):
    
    print('Creating generator...')
    test_generator = SequenceGenerator(img_dir, img_sources, n_timesteps,
                                       frame_step=frame_step, seq_overlap=5, 
                                       max_seq_per_video=max_seq_per_video, 
                                       N_seq=N_seq, shuffle=shuffle, 
                                       batch_size=batch_size,
                                       max_missing_frames=max_missing_frames, 
                                       seed=seed, data_format=data_format)
    
    n = 0
    in_memory_ratio = 20
    X = []
    X_hat = []
    #mse_model = 0
    #mse_prev = 0
    
    n_batches = ((len(test_generator.possible_starts) - 1) // batch_size) + 1 # ceil
    print('Number of sequences: {}'.format(len(test_generator.possible_starts)))
    print('Number of batches: {}'.format(n_batches))

    for i in tqdm(range(n_batches)):
        X_, y = next(test_generator)
        pred = test_model.predict(X_, batch_size)

        #mse_model += np.mean((X[:, 1:] - pred[:, 1:]) ** 2)  # look at all timesteps except the first
        #mse_prev += np.mean((X[:, :-1] - X[:, 1:]) ** 2)

        if n % in_memory_ratio == 0:
            X.extend(X_)
            X_hat.extend(pred)

        n += 1

    X = np.array(X)
    X_hat = np.array(X_hat)

    if data_format == 'channels_first':
        X = np.transpose(X, (0, 1, 3, 4, 2))
        X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

    #mse_model /= (n * batch_size)
    #mse_prev /= (n * batch_size)
    
    return X, X_hat
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pre-trained model.')
    parser.add_argument('config', help='experiment config name defined in moments_setting.py')
    FLAGS, unparsed = parser.parse_known_args()
    
    experiment = experiments[FLAGS.config]
    print('\n==> Starting experiment: {}\n'.format(experiment['description']))
    
    print('Loading pre-trained model...')
    pretrained_model = load_model(**experiment)
    layer_config = pretrained_model.layers[1].get_config()
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    
    print('Creating testing model...')
    test_model = create_test_model(pretrained_model, **experiment)
    
    X, X_hat = evaluate(data_format=data_format, **experiment)
    
    if experiment['output_mode'] == 'prediction':
        save_predictions(X, X_hat, FLAGS.config, **experiment)
