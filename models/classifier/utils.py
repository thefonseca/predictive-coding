import numpy as np
from skimage.transform import resize
import os


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

def get_create_results_dir(config_name, base_results_dir):
    results_dir = os.path.join(base_results_dir, config_name)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    return results_dir