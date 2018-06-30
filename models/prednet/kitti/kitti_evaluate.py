'''
Evaluate trained PredNet on KITTI sequences.
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
from keras.utils import multi_gpu_model

from prednet import PredNet
from kitti_data import SequenceGenerator
from kitti_settings import *
from tqdm import tqdm

n_plot = 40
batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)
#test_model = multi_gpu_model(test_model, gpus=4)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
#X_hat = test_model.predict(X_test, batch_size, verbose=1)

# Alternative implementation to check the time spent for each batch
n = 0
X_hat = []
mse_model = 0
mse_prev = 0

for idx in tqdm(range(0, X_test.shape[0], batch_size)):
    idx_end = idx + batch_size
    if idx_end > X_test.shape[0]:
        idx_end = X_test.shape[0]
        
    X = X_test[idx:idx_end]
    pred = test_model.predict(X, batch_size)
    X_hat.extend(pred)
    
    mse_model += np.mean((X[:, 1:] - pred[:, 1:]) ** 2)  # look at all timesteps except the first
    mse_prev += np.mean((X[:, :-1] - X[:, 1:]) ** 2)
    
    n += 1
    
X_hat = np.array(X_hat)
print(X_hat.shape)
#X_hat = np.reshape(X_hat, (-1,) + X_hat.shape[2:])
#print(X_hat.shape)

mse_model /= (n * batch_size)
mse_prev /= (n * batch_size)
    
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
#mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
#mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.makedirs(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
    plt.clf()
