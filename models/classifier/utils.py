import numpy as np
from skimage.transform import resize
import os


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

def get_create_results_dir(config_name, base_results_dir):
    results_dir = os.path.join(base_results_dir, config_name)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    return results_dir

def get_config_str(config):
    config_str = ''
    for k, v in sorted(config.items()):
        if k != 'description':
            config_str += '    {}: {}\n'.format(k, v)
    return config_str

def get_config(configs, tasks, FLAGS):
    config = configs[FLAGS['config']]
    
    if not FLAGS['task']:
        suffix = '__' + config['task']
    else:
        suffix = '__' + FLAGS['task']
        config['task'] = FLAGS['task']
        
    if not FLAGS['model_type']:
        prefix = config['model_type'] + '__'
    else:
        prefix = FLAGS['model_type'] + '__'
        config['model_type'] = FLAGS['model_type']
        
    config.update(tasks[config['task']])
    
    name = prefix + FLAGS['config'] + suffix
    config['_config_name'] = name
    config['_config_name_original'] = FLAGS['config']
    return name, config