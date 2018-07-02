'''
Evaluate trained PredNet on Moments in Time sequences.
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

from prednet import PredNet
from settings import configs
import utils
import sys
sys.path.append("../classifier")
from data import DataGenerator

from tqdm import tqdm
import argparse
import csv
import cPickle as pkl
from skimage.measure import block_reduce

FLAGS = None


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
        
def save_representation(rep, sources, results_dir, config):
    
    for i, label in enumerate(sources):
        target_dir = results_dir
        if len(label) > 1:
            category, source = label
            target_dir = os.path.join(results_dir, category)
        else:
            source = label[0]
        
        if not os.path.exists(target_dir): os.makedirs(target_dir)
            
        features_file = '{}.pkl'.format(source)
        filename = os.path.join(target_dir, rep_file)
        
        with open(filename, 'w') as f:
            pkl.dump(rep[i].reshape(rep.shape[2:]), f)
        
def evaluate_prediction(model, dataset, experiment_name, 
                        data_generator, n_batches, base_results_dir,
                        data_format=K.image_data_format(), **config):
    
    n = 0
    in_memory_ratio = 20
    X = []
    preds = []
    mse_model = 0
    mse_prev = 0
    
    for i in tqdm(range(n_batches)):
        X_, y_, _ = next(data_generator)
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
        
    results_dir = utils.get_create_results_dir(experiment_name, 
                                               base_results_dir, 
                                               dataset=dataset)
    save_predictions(X, preds, mse_model, mse_prev, results_dir, **config)
    
    
    
def evaluate_representation(model, dataset, experiment_name, output_mode, 
                            data_generator, n_batches, base_results_dir,
                            timestep_start=-1, timestep_end=None,
                            data_format=K.image_data_format(), **config):
    
    results_dir = utils.get_create_results_dir(experiment_name, 
                                               base_results_dir, 
                                               dataset=dataset)
    sources = []
    data_iterator = iter(data_generator)
    for i in tqdm(range(n_batches)):
        X_, y_, sources_ = next(data_iterator)
        rep = model.predict(X_, data_generator.batch_size)
        
        rep = rep[:, timestep_start:timestep_end]
        sources_ = sources_[:, timestep_start:timestep_end].flatten()
        source_batch = []
        
        for s in sources_:
            path, source = os.path.split(s)
            path, category = os.path.split(path)
            path, data_split = os.path.split(path)
            source = source.replace('.jpg', '')
            category_source = (category, source)
            source_batch.append(category_source)
            sources.append(category_source)
        
        if output_mode == 'representation':
            rep_layers = model.layers[1].unflatten_features(X_.shape, rep)
            rep = []
            
            width_index = 3
            channel_index = -1
            block_size = [1, 1, 1, 1, 1]
            if data_format == 'channels_first':
                width_index = -1
                channel_index = 2
            
            for i, rep_layer in enumerate(rep_layers): 
                # Do avg spatial pooling to make all representation layers have 
                # the same dimension as the higher-level representation layer.
                ratio = rep_layer.shape[width_index] / rep_layers[-1].shape[width_index]
                if ratio > 1:
                    block_size[width_index-1] = ratio
                    block_size[width_index] = ratio
                    rep_layer = block_reduce(rep_layer, block_size=tuple(block_size), 
                                             func=np.mean)
                rep.append(rep_layer)
            rep = np.concatenate(rep, axis=channel_index)
            
        if data_format == 'channels_first':
            rep = np.transpose(rep, (0, 1, 3, 4, 2))
            
        save_representation(rep, source_batch, results_dir, config)
    
    # Save labels in csv file
    f = os.path.join(results_dir, 'sources.csv')
    with open(f, 'wb') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['category','source'])
        for row in sources:
            csv_out.writerow(row)

            
def evaluate(config_name, dataset, data_dir, output_mode, 
             classes=None, n_timesteps=10, frame_step=3, 
             seq_overlap=0, input_width=160, input_height=128, 
             shuffle=False, batch_size=5, 
             stateful=False, rescale=None, **config):
    
    model = utils.create_model(train=False, stateful=stateful, 
                               batch_size=batch_size, 
                               output_mode=output_mode, **config)
    model.summary()
    
    layer_config = model.layers[1].get_config()
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    
    print('Creating generator...')
    resize = lambda img: utils.resize_img(img, target_size=(input_height, 
                                                            input_width))
    
    data_generator = DataGenerator(classes=classes,
                                   seq_length=n_timesteps,
                                   seq_overlap=seq_overlap,
                                   sample_step=frame_step,
                                   target_size=None, #input_shape,
                                   rescale=rescale,
                                   fn_preprocess=resize,
                                   batch_size=batch_size, 
                                   shuffle=shuffle,
                                   return_sources=True,
                                   data_format=data_format)
    
    data_generator = data_generator.flow_from_directory(data_dir)
    n_batches = len(data_generator)
    print('Number of batches: {}'.format(n_batches))
    
    if output_mode == 'prediction':
        evaluate_prediction(model, dataset, config_name, 
                            data_generator, n_batches, 
                            data_format=data_format, **config)
        
    elif output_mode == 'representation' or output_mode[:1] == 'R':
        evaluate_representation(model, dataset, config_name, output_mode, 
                                data_generator, n_batches,
                                data_format=data_format, **config)
        
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PredNet model.')
    parser.add_argument('config', help='experiment config name defined in moments_settings.py')
    FLAGS, unparsed = parser.parse_known_args()
    
    config = configs[FLAGS.config]
    
    print('\n==> Starting experiment: {}\n'.format(config['description']))
    config_str = utils.get_config_str(config)
    print('==> Using configuration:\n{}'.format(config_str))
    
    for split in ['training', 'validation', 'test']:
        img_dir = config.get(split + '_data_dir', None)
        img_sources = config.get(split + '_data_sources', None)
        
        if img_dir and img_sources:
            print('==> Dataset split: {}'.format(split))
            evaluate(FLAGS.config, split, img_dir, **config)
            utils.save_experiment_config(FLAGS.config, config['base_results_dir'], 
                                         config, dataset=split)
    