import os
import argparse

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
#from keras.models import load_model
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.losses import categorical_crossentropy

from data import DataGenerator
from settings import configs, tasks
import models
import utils

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import random as rn

# Getting reproducible results:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def save_experiment_config(config_name, base_results_dir, config):
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    f = open(os.path.join(results_dir, 'experiment_config.txt'), 'w')
    for key in sorted(config):
        f.write('{}: {}\n'.format(key, config[key]))
    f.close()
    
def resize_fn(input_height, input_width):
    resize = None
    if input_width and input_height:
        resize = lambda img: utils.resize_img(img, target_size=(input_height, 
                                                                input_width))
    return resize
    
def train(config_name, training_data_dir, validation_data_dir, 
          base_results_dir, hidden_dims, test_data_dir=None, 
          epochs=10, use_multiprocessing=False, workers=1, 
          dropout=0.5, seq_length=None, sample_step=1, 
          batch_size=10, stopping_patience=3, classes=None, 
          max_queue_size=10, model_type='convnet', shuffle=True,
          training_index_start=0, training_max_per_class=None, 
          validation_index_start=0, validation_max_per_class=None, 
          input_width=None, input_height=None, rescale=None,
          min_seq_length=0, pad_sequences=False, max_seq_per_source=None,
          data_format=K.image_data_format(), **config):
    
    train_generator = DataGenerator(batch_size=batch_size,
                                    shuffle=shuffle,
                                    classes=classes,
                                    seq_length=seq_length,
                                    min_seq_length=min_seq_length,
                                    max_seq_per_source=max_seq_per_source,
                                    pad_sequences=pad_sequences,
                                    sample_step=sample_step,
                                    rescale=rescale,
                                    data_format=data_format,
                                    fn_preprocess=resize_fn(input_height, 
                                                            input_width),
                                    index_start=training_index_start,
                                    max_per_class=training_max_per_class)
    
    val_generator = DataGenerator(batch_size=batch_size,
                                  shuffle=shuffle,
                                  classes=classes,
                                  seq_length=seq_length,
                                  min_seq_length=min_seq_length,
                                  max_seq_per_source=max_seq_per_source,
                                  pad_sequences=pad_sequences,
                                  sample_step=sample_step,
                                  rescale=rescale,
                                  data_format=data_format,
                                  fn_preprocess=resize_fn(input_height, 
                                                          input_width),
                                  index_start=validation_index_start,
                                  max_per_class=validation_max_per_class)
    
    train_generator = train_generator.flow_from_directory(training_data_dir)
    val_generator = val_generator.flow_from_directory(validation_data_dir)
    
    if len(train_generator) == 0 or len(val_generator) == 0:
        return
    
    input_shape = train_generator.data_shape
    n_classes = train_generator.n_classes
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    
    model_fn = getattr(models, model_type)
    mask_value = 0. if pad_sequences else None
    model = model_fn(input_shape, n_classes, hidden_dims,
                     drop_rate=dropout, mask_value=mask_value, **config)
    checkpoint_path = os.path.join(results_dir, model_type + '.hdf5')
    csv_path = os.path.join(results_dir, model_type + '.log')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    json_file = os.path.join(results_dir, model_type + '.json')
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
    
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, 
                                   verbose=1, save_best_only=True)
    
    csv_logger = CSVLogger(csv_path)
    stopper = EarlyStopping(monitor='val_loss', 
                            patience=stopping_patience, 
                            verbose=0, mode='auto')
    
    model.fit_generator(train_generator,
                        len(train_generator),
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        callbacks=[checkpointer, csv_logger, stopper],
                        use_multiprocessing=use_multiprocessing,
                        max_queue_size=max_queue_size, 
                        workers=workers)
    
def gen_multiple(generators):
    while True:
        X_list = [gen.next()[0] for gen in generators]
        y = generators[0].next()[1]
        yield X_list, y
        
def evaluate_average(model, data_iterator, n_batches):
    predictions = {}
    source_counts = {}
    labels = {}

    for i in tqdm(range(n_batches), desc='Evaluating on test set'):
        X, y, sources_ = next(data_iterator)
        sources = []
        for s in sources_:
            path, source = os.path.split(s[0])
            path, category = os.path.split(path)
            source = source.split('__')
            source = '__'.join(source[:-1])
            sources.append(os.path.join(category, source))

        preds = model.predict(X)

        for j in range(len(sources)):
            s_count = source_counts.get(sources[j], 0)
            source_counts[sources[j]] = s_count + 1
            acc_pred = predictions.get(sources[j], np.zeros_like(y[j]))
            predictions[sources[j]] = acc_pred + preds[j]
            labels[sources[j]] = y[j]

    metrics = {}
    y_true = np.array([y for source, y in sorted(labels.items())])
    y_pred = np.array([(1. * predictions[s]) / source_counts[s] for s in sorted(predictions.keys())])

    predictions = { 'y_true': y_true, 'y_pred': y_pred, 
                    'sources': sorted(labels.keys()) }

    with open(os.path.join(results_dir, 'predictions.pkl'), 'w') as f:
        pkl.dump(predictions, f)

    y_true = K.variable(y_true)
    y_pred = K.variable(y_pred)

    loss = categorical_crossentropy(y_true, y_pred)
    metrics['loss'] = K.eval(K.mean(loss))
    acc = categorical_accuracy(y_true, y_pred)
    metrics['acc'] = K.eval(K.mean(acc))

    if y_true.shape[-1] > 10:
        top_k_acc = top_k_categorical_accuracy(y_true, y_pred, k=5)
        metrics['top_k_acc'] = K.eval(top_k_acc)
    return metrics


def evaluate(config_name, test_data_dir, hidden_dims,
             base_results_dir, classes=None, sample_step=1,
             workers=1, use_multiprocessing=False, max_seq_per_source=None,
             seq_length=None, min_seq_length=0, pad_sequences=False,
             test_max_per_class=None, test_index_start=0,
             model_type='convnet', average_predictions=False, 
             input_height=None, input_width=None, rescale=None,
             data_format=K.image_data_format(), **config):
    
    print('\nEvaluating model on test set...')
    # we use the remaining part of training set as test set
    print('Classes: {}'.format(classes))
    generator = DataGenerator(classes=classes,
                              batch_size=1,
                              sample_step=sample_step,
                              seq_length=seq_length,
                              min_seq_length=min_seq_length,
                              max_seq_per_source=max_seq_per_source,
                              pad_sequences=pad_sequences,
                              return_sources=average_predictions,
                              data_format=data_format,
                              rescale=rescale,
                              fn_preprocess=resize_fn(input_height, 
                                                      input_width),
                              index_start=test_index_start,
                              max_per_class=test_max_per_class)
    generator = generator.flow_from_directory(test_data_dir)
    
    if len(generator) == 0:
        return
    
    input_shape = generator.data_shape
    n_classes = generator.n_classes
    config['batch_size'] = generator.batch_size
    
    # load best model
    results_dir = utils.get_create_results_dir(config_name, base_results_dir)
    checkpoint_path = os.path.join(results_dir, model_type + '.hdf5')       
    #model = load_pretrained(checkpoint_path)
    model_fn = getattr(models, model_type)
    mask_value = 0. if pad_sequences else None
    model = model_fn(input_shape, n_classes, hidden_dims,
                     drop_rate=0, mask_value=mask_value, **config)
    checkpoint_path = os.path.join(results_dir, model_type + '.hdf5')
    model.load_weights(checkpoint_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['categorical_accuracy']) 
    model.summary()
    
    if average_predictions:
        # Average predictions for sequences coming from the 
        # same source video
        n_batches = len(generator)
        metrics = evaluate_average(model, iter(generator), n_batches)
        metric_str = ['{}: {}'.format(m, v) for m, v in metrics.items()]
        metric_str = ' - '.join(metric_str)
    else:
        metrics = model.evaluate_generator(generator, len(generator),
                                           use_multiprocessing=use_multiprocessing, 
                                           workers=workers)
        metric_str = ['{}: {}'.format(m, v) for m, v in zip(model.metrics_names, metrics)]
        metric_str = ' - '.join(metric_str)
    
    print('Test {}'.format(metric_str))
    f = open(os.path.join(results_dir, 'test.txt'), 'w')
    f.write('Test results:\n')
    f.write(metric_str)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier.')
    parser.add_argument('config', help='experiment config name defined in settings.py')
    parser.add_argument('-m', '--model_type', type=str, help='model architecture of classifier')
    parser.add_argument('-t', '--task', type=str, choices=['2c_easy', '2c_hard', '10c'],
                    help='classification task')
    parser.add_argument('--eval', help='perform only evaluation using pretrained model',
                        action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', help='list of gpus to use')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    config_name, config = utils.get_config(vars(FLAGS))
    
    if FLAGS.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in FLAGS.gpus])
    
    print('\n==> Starting experiment: {}'.format(config['description']))
    config_str = utils.get_config_str(config)
    print('\n==> Using configuration:\n{}'.format(config_str))
    
    if not FLAGS.eval:
        train(config_name, **config)
        save_experiment_config(config_name, config['base_results_dir'], config)
    
    if config.get('test_data_dir', None) or config.get('ensemble', None):
        evaluate(config_name, **config)

    
