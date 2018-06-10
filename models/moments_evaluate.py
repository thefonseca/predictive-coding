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

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from moments_data import SequenceGenerator
from moments_settings import *
from tqdm import tqdm


def load_model(json_file, weights_file):
    # Load trained model
    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    train_model.load_weights(weights_file)
    return train_model

def create_test_model(train_model, output_mode, n_timesteps=10):
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
    return test_model

def save_predictions(save_dir, output_mode, X, X_hat, 
                     n_plot=20, experiment_name='exp'):
    
    # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
    mse_model = np.mean((X[:, 1:] - X_hat[:, 1:]) ** 2)  # look at all timesteps except the first
    mse_prev = np.mean((X[:, :-1] - X[:, 1:]) ** 2)
    
    save_dir = os.path.join(save_dir, experiment_name + '_' + output_mode)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    f = open(os.path.join(save_dir, 'prediction_scores.txt'), 'w')
    f.write("Model MSE: %f\n" % mse_model)
    f.write("Previous Frame MSE: %f" % mse_prev)
    f.close()

    # Plot some predictions
    n_timesteps =  X.shape[1]
    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
    plt.figure(figsize = (n_timesteps, 2 * aspect_ratio))
    gs = gridspec.GridSpec(2, n_timesteps)
    gs.update(wspace=0., hspace=0.)
    plot_save_dir = os.path.join(save_dir, 'prediction_plots/')
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
             seed=17, data_format=K.image_data_format()):
    
    print('Creating generator...')
    test_generator = SequenceGenerator(img_dir, img_sources, n_timesteps,
                                       frame_step=frame_step, seq_overlap=5, 
                                       max_seq_per_video=max_seq_per_video,
                                       shuffle=shuffle, batch_size=batch_size,
                                       max_missing_frames=max_missing_frames, 
                                       seed=seed, data_format=data_format)
    
    n = 0
    in_memory_ratio = 20
    X = []
    X_hat = []
    #mse_model = 0
    #mse_prev = 0

    for X_, y in tqdm(test_generator):
        pred = test_model.predict(X_, batch_size)

        #mse_model += np.mean((X[:, 1:] - pred[:, 1:]) ** 2)  # look at all timesteps except the first
        #mse_prev += np.mean((X[:, :-1] - X[:, 1:]) ** 2)

        if n % in_memory_ratio == 0:
            X.extend(X_)
            X_hat.extend(pred)

        n += 1

        #if n > 2:
        #    break

    X = np.array(X)
    X_hat = np.array(X_hat)
    #X_test = np.reshape(X_test, (-1,) + X_test.shape[2:])
    #X_hat = np.reshape(X_hat, (-1,) + X_hat.shape[2:])

    if data_format == 'channels_first':
        X = np.transpose(X, (0, 1, 3, 4, 2))
        X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

    #mse_model /= (n * batch_size)
    #mse_prev /= (n * batch_size)
    
    return X, X_hat
    
    
if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Evaluate pre-trained model.')
    #parser.add_argument('config', help='experiment config name defined in ')
    #FLAGS, unparsed = parser.parse_known_args()
    
    #TODO: store configuration in separate file
    
    output_mode = 'prediction' # 'prediction' | 'error' | 'features'
    n_timesteps = 10
    
    weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    
    print('Loading pre-trained model...')
    pretrained_model = load_model(json_file, weights_file)
    layer_config = pretrained_model.layers[1].get_config()
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    
    print('Creating testing model...')
    test_model = create_test_model(pretrained_model, output_mode, n_timesteps=n_timesteps)
    
    img_dir = os.path.join(DATA_DIR, 'test')
    img_sources = os.path.join(DATA_DIR, 'sources_test.pkl')
    
    X, X_hat = evaluate(img_dir, img_sources, output_mode, n_timesteps=n_timesteps, 
                        frame_step=3, seq_overlap=5, max_seq_per_video=5, 
                        shuffle=False, batch_size=5, max_missing_frames=15, 
                        seed=17, data_format=data_format)
    
    if output_mode == 'prediction':
        save_predictions(RESULTS_SAVE_DIR, output_mode, X, X_hat, 
                         n_plot=20, experiment_name='moments')
