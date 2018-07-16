import os
import argparse

from data import DataGenerator
import utils

from tqdm import tqdm
import numpy as np
import random as rn
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

#from keras import backend as K
#from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
#from keras.losses import categorical_crossentropy

# Getting reproducible results:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

def save_experiment_config(config_name, base_results_dir, config):
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    f = open(os.path.join(results_dir, 'experiment_config.txt'), 'w')
    for key in sorted(config):
        f.write('{}: {}\n'.format(key, config[key]))
    f.close()
    
def train(config_name, training_data_dir, base_results_dir, classes=None, 
          training_index_start=0, training_max_per_class=None, 
          model_type='linear', **config):
    
    train_generator = DataGenerator(batch_size=1, classes=classes,
                                    max_per_class=training_max_per_class)
    train_generator = train_generator.flow_from_directory(training_data_dir)
    train_iterator = iter(train_generator)

    if len(train_generator) == 0:
        return
    
    train_X = []
    train_y = []
    for i in tqdm(range(len(train_generator)), desc='Loading training set'):
        X, y = next(train_iterator)
        train_X.append(X.flatten())
        train_y.append(y[0][0])
    
    print('Training linear model...')
    clf = LinearSVC()
    #clf = LogisticRegression()
    clf.fit(train_X, train_y)
    
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    model_path = os.path.join(results_dir, model_type + '.pkl')
    joblib.dump(clf, model_path)
    
def evaluate_average(model, data_iterator, n_batches):
    predictions = {}
    source_counts = {}
    labels = {}
    y_pred = []
    y_true = []

    for i in tqdm(range(n_batches), desc='Evaluating on test set'):
        X, y, sources_ = next(data_iterator)
        sources = []
        for s in sources_:
            path, source = os.path.split(s)
            path, category = os.path.split(path)
            source = source.split('__')
            source = '__'.join(source[:-1])
            sources.append(os.path.join(category, source))
            
        preds = model.predict([X_.flatten() for X_ in X])
        y_pred.extend(preds)
        y_true.extend([np.argmax(y_) for y_ in y])
        
        for j in range(len(sources)):
            s_count = source_counts.get(sources[j], 0)
            source_counts[sources[j]] = s_count + 1
            acc_pred = predictions.get(sources[j], np.zeros_like(preds[j]))
            predictions[sources[j]] = acc_pred + preds[j]
            labels[sources[j]] = np.argmax(y[j])

    y_avg_true = np.array([y for source, y in sorted(labels.items())])
    y_avg_pred = np.array([1 if predictions[s] / source_counts[s] >= 0.5 else 0 for s in sorted(predictions.keys())])
    
    metrics = {}
    acc = accuracy_score(y_true, y_pred)
    metrics['acc'] = acc
    acc = accuracy_score(y_avg_true, y_avg_pred)
    metrics['avg_acc'] = acc
    return metrics

def evaluate(config_name, test_data_dir, base_results_dir,
             average_predictions=True, classes=None,
             test_index_start=0, test_max_per_class=None,
             model_type='linear', **config):
    
    test_generator = DataGenerator(batch_size=1,
                                   classes=classes,
                                   return_sources=True,
                                   index_start=test_index_start,
                                   max_per_class=test_max_per_class)
    test_generator = test_generator.flow_from_directory(test_data_dir)
    test_iterator = iter(test_generator)
    
    if len(test_generator[0]) == 0:
        return
        
    # load model
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    model_path = os.path.join(results_dir, model_type + '.pkl')
    model = joblib.load(model_path)
    
    if average_predictions:
        # Average predictions for sequences coming from the 
        # same source video
        n_batches = len(test_generator)
        metrics = evaluate_average(model, iter(test_generator), n_batches)
        metric_str = ['{}: {}'.format(m, v) for m, v in metrics.items()]
        metric_str = ' - '.join(metric_str)
    else:
        preds = svm.predict(test_X)
        acc = accuracy_score(test_y, preds)
        metric_str = 'acc: {}'.format(acc)
        
    print('Test {}'.format(metric_str))
    f = open(os.path.join(results_dir, 'test.txt'), 'w')
    f.write('Test results:\n')
    f.write(metric_str)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier.')
    parser.add_argument('config', help='experiment config name defined in settings.py')
    parser.add_argument('-t', '--task', type=str, choices=['2c_easy', '2c_hard', '10c'],
                    help='classification task')
    parser.add_argument('--eval', help='perform only evaluation using pretrained model',
                        action='store_true')
    
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS_dict = vars(FLAGS)
    FLAGS_dict['model_type'] = 'linear'
    config_name, config = utils.get_config(FLAGS_dict)
    
    print('\n==> Starting experiment: {}'.format(config['description']))
    config_str = utils.get_config_str(config)
    print('\n==> Using configuration:\n{}'.format(config_str))
    
    if not FLAGS.eval:
        train(config_name, **config)
        save_experiment_config(config_name, config['base_results_dir'], config)
    
    if config.get('test_data_dir', None):
        evaluate(config_name, **config)

    