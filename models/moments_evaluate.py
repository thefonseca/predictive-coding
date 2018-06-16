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
import csv
import cPickle as pkl
#import gzip
import bz2

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

def get_create_results_dir(experiment_name, output_mode, config):
    results_dir = os.path.join(config['base_results_dir'], experiment_name + '_' + output_mode)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    return results_dir

def save_experiment_config(experiment_name, output_mode, config):
    
    results_dir = get_create_results_dir(experiment_name, output_mode, config)
    f = open(os.path.join(results_dir, 'experiment_config.txt'), 'w')
    
    for key in sorted(config):
        f.write('{}: {}\n'.format(key, config[key]))
        
    f.close()

def save_predictions(X, X_hat, mse_model, mse_prev, results_dir, 
                     n_plot=20, **config):
    
    # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
    #mse_model = np.mean((X[:, 1:] - X_hat[:, 1:]) ** 2)  # look at all timesteps except the first
    #mse_prev = np.mean((X[:, :-1] - X[:, 1:]) ** 2)
    
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
        
def save_representation(rep, labels, results_dir, config):
    
    for i, label in enumerate(labels):
        
        target_dir = results_dir
        if isinstance(label, tuple):
            category, source = label
            target_dir = os.path.join(results_dir, category)
        
        if not os.path.exists(target_dir): os.makedirs(target_dir)
            
        rep_file = '{}__{:03d}.pkl'.format(source, i)
        filename = os.path.join(target_dir, rep_file)
        
        #with gzip.GzipFile(filename, 'w') as f:
        with bz2.BZ2File(filename, 'w') as f:
            pkl.dump(rep[i].reshape(rep.shape[2:]), f)
        #pkl.dump(rep, open(filename, "wb"))
    
        
def evaluate_prediction(model, experiment_name, output_mode, 
                        data_generator, n_batches, 
                        data_format=K.image_data_format(), **config):
    
    n = 0
    in_memory_ratio = 20
    X = []
    preds = []
    mse_model = 0
    mse_prev = 0
    
    for i in tqdm(range(n_batches)):
        X_, y_ = next(data_generator)
        pred = model.predict(X_, data_generator.batch_size)

        mse_model += np.mean((X_[:, 1:] - pred[:, 1:]) ** 2)  # look at all timesteps except the first
        mse_prev += np.mean((X_[:, :-1] - X_[:, 1:]) ** 2)
        
        if n % in_memory_ratio == 0:
            X.extend(X_)
            preds.extend(pred)
            
        n += 1

    X = np.array(X)
    preds = np.array(preds)

    mse_model /= (n * data_generator.batch_size)
    mse_prev /= (n * data_generator.batch_size)  
            
    if data_format == 'channels_first':
        X = np.transpose(X, (0, 1, 3, 4, 2))
        preds = np.transpose(preds, (0, 1, 3, 4, 2))
        
    results_dir = get_create_results_dir(experiment_name, output_mode, config)
    save_predictions(X, preds, mse_model, mse_prev, results_dir, **config)
    
    
    
def evaluate_representation(model, experiment_name, output_mode, 
                            data_generator, n_batches, 
                            timestep_start=-1, timestep_end=None,
                            data_format=K.image_data_format(), **config):
    
    results_dir = get_create_results_dir(experiment_name, output_mode, config)
    y = []

    for i in tqdm(range(n_batches)):
        X_, y_ = next(data_generator)
        rep = model.predict(X_, data_generator.batch_size)
        
        rep = rep[:, timestep_start:timestep_end]
        y_ = y_[:, timestep_start:timestep_end].flatten()
        
        category_source = [tuple(label.split('__')) for label in y_]
        y.extend(category_source)
        
        if output_mode == 'representation':
            rep = model.get_layer('prednet_1').unflatten_features(X_.shape, rep)

            if data_format == 'channels_first':
                for f in preds: 
                    f = np.transpose(f, (0, 1, 3, 4, 2))
                    #print(f.shape)

        elif data_format == 'channels_first':
            rep = np.transpose(rep, (0, 1, 3, 4, 2))
        
        save_representation(rep, category_source, results_dir, config)
    
    # Save labels in csv file
    f = os.path.join(results_dir, 'labels.csv')
    with open(f, 'wb') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['category','source'])
        for row in y:
            csv_out.writerow(row)


def evaluate(model, experiment_name, img_dir, img_sources, 
             output_mode, n_timesteps=10, frame_step=3, seq_overlap=5, 
             max_seq_per_video=5, shuffle=False, batch_size=5, 
             max_missing_frames=15, N_seq=None, seed=17, 
             data_format=K.image_data_format(), **config):
    
    print('Creating generator...')
    data_generator = SequenceGenerator(img_dir, img_sources, n_timesteps,
                                       output_mode=output_mode,
                                       frame_step=frame_step, seq_overlap=5, 
                                       max_seq_per_video=max_seq_per_video, 
                                       N_seq=N_seq, shuffle=shuffle, 
                                       batch_size=batch_size,
                                       max_missing_frames=max_missing_frames, 
                                       seed=seed, data_format=data_format)
    
    n_batches = ((len(data_generator.possible_starts) - 1) // batch_size) + 1 # ceil
    print('Number of sequences: {}'.format(len(data_generator.possible_starts)))
    print('Number of batches: {}'.format(n_batches))
    
    if output_mode == 'prediction':
        evaluate_prediction(model, experiment_name, output_mode, 
                            data_generator, n_batches, 
                            data_format=data_format, **config)
        
    elif output_mode == 'representation' or output_mode[:1] == 'R':
        evaluate_representation(model, experiment_name, output_mode, 
                                data_generator, n_batches,
                                data_format=data_format, **config)
        
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pre-trained model.')
    parser.add_argument('config', help='experiment config name defined in moments_setting.py')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prediction', help='model will output image predictions', 
                       action="store_true")
    group.add_argument('--error', help='model will output prediction errors', 
                       action="store_true")
    group.add_argument('--representation', help='model will output PredNet representation (R) for all layers', 
                       action="store_true")
    parser.add_argument('--layer', help='output specific layer number', type=int)
    FLAGS, unparsed = parser.parse_known_args()
    
    experiment = experiments[FLAGS.config]
    
    for arg in ['prediction', 'representation', 'error']:
        if getattr(FLAGS, arg):
            output_mode = arg
            
    if FLAGS.layer is not None:
        prednet_output_mode = output_mode[:1].upper() + str(FLAGS.layer)
    else:
        prednet_output_mode = output_mode
        
    print(prednet_output_mode)
        
    print('\n==> Starting experiment: {}'.format(experiment['description']))
    if FLAGS.layer is None:
        print('==> Output mode: {}\n'.format(output_mode))
    else:
        print('==> Output mode: {} (layer {})\n'.format(output_mode, FLAGS.layer))
    
    print('Loading pre-trained model...')
    pretrained_model = load_model(**experiment)
    layer_config = pretrained_model.layers[1].get_config()
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    
    print('Creating testing model...')
    model = create_test_model(pretrained_model, prednet_output_mode, **experiment)
    model.summary()
    
    evaluate(model, FLAGS.config, output_mode=prednet_output_mode, 
             data_format=data_format, **experiment)
    
    save_experiment_config(FLAGS.config, prednet_output_mode, experiment)
    
    
