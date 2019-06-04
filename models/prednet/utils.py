import os
import numpy as np
from skimage.transform import resize

from settings import configs, tasks


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
    x,y,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)   
    return img[startx:startx+cropx,starty:starty+cropy]


def resize_img(img, target_size):
    ratios = [float(target_size[i]) / img.shape[i] for i in range(len(target_size))]
    larger_ratio = max(ratios)
    new_shape = [float(larger_ratio) * img.shape[i] for i in range(len(target_size))]
    
    img = resize(img, (int(np.round(new_shape[0])), 
                       int(np.round(new_shape[1]))),
                 mode='reflect')
    
    # crop
    img = crop_center(img, target_size[0], target_size[1])
    return img


def get_config(FLAGS):
    config = dict()
    config.update(configs[FLAGS['config']])
    
    if FLAGS.get('task', None) is None:
        task_suffix = '__' + config['task']
    else:
        task_suffix = '__' + FLAGS['task']
        config['task'] = FLAGS['task']
    
    if FLAGS.get('stateful', None) is None:
        stateful = config['stateful']
    else:
        stateful = FLAGS['stateful']
        config['stateful'] = FLAGS['stateful']
    
    if FLAGS.get('pretrained', None) is None:
        model_suffix = '__' + config['pretrained']
    else:
        model_suffix = '__' + FLAGS['pretrained']
        config['pretrained'] = FLAGS['pretrained']
        
    if config['model_weights_file']:
        config['model_weights_file'] = config['model_weights_file'].format(model_suffix)
    if config['model_json_file']:
        config['model_json_file'] = config['model_json_file'].format(model_suffix)
        
    if config['output_mode'] in ['prediction', 'representation']:
        name = FLAGS['config'] + model_suffix
    else:
        name = FLAGS['config'] + task_suffix
        
    if stateful:
        name += '__' + 'stateful'

    if FLAGS.get('train_dir'):
        config['training_data_dir'] = FLAGS.get('train_dir')

    if FLAGS.get('val_dir'):
        config['validation_data_dir'] = FLAGS.get('val_dir')
    
    config['_config_name'] = name
    config['_config_name_original'] = FLAGS['config']
    if config['task'] in tasks:
        config.update(tasks[config['task']])
    else:
        config['classes'] = None
    return name, config