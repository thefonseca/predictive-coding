import os

DATA_DIR = '../../datasets/moments_video_frames/'
FRAMES_PER_VIDEO = 90
SEQUENCES_PER_VIDEO = 5

def add_config(configs, name, config, base_config=None):
    new_config = dict()
    if base_config:
        new_config.update(base_config)       
    new_config.update(config)
    new_config.update(tasks[new_config['task']])
    new_config.update(models[new_config['model_name']])
    configs[name] = new_config
    

configs = {}

tasks = {
    '3c': {
        'classes': ['cooking', 'running', 'walking']
    },
    '10c': {
        'classes': ['barking', 'cooking', 'driving', 'juggling', 'photographing', 
                    'biting', 'climbing', 'running', 'sleeping', 'walking']
    }
}

models = {
    'prednet_kitti': {
        'model_weights_file': './kitti/model_data/kitti_keras/prednet_kitti_weights.hdf5',
        'model_json_file': './kitti/model_data/kitti_keras/prednet_kitti_model.json',
    },
    'prednet_random': {
        'model_weights_file': None,
        'model_json_file': None
    },
    'prednet_kitti_finetuned_moments': {
        'model_weights_file': './results/prednet_kitti__moments__model{}/weights.hdf5',
        'model_json_file': './results/prednet_kitti__moments__model__{}/model.json',
    },
    'prednet_random_finetuned_moments': {
        'model_weights_file': './results/prednet_kitti__moments__model{}/weights.hdf5',
        'model_json_file': './results/prednet_kitti__moments__model{}/model.json',
    }
}

eval_base_config = {
    'n_timesteps': 10,
    'frame_step': 3,
    'seq_overlap': 5,
    'timestep_start': -1,
    'batch_size': 20,
    'stateful': False,
    'input_channels': 3, 
    'input_height': 128, 
    'input_width': 160,
    'rescale': 1./255,
    'shuffle': False,
    'workers': 4,
    #'use_multiprocessing': True,
    # DATA
    'training_data_dir': os.path.join(DATA_DIR, 'training'),
    'validation_data_dir': os.path.join(DATA_DIR, 'validation'),
    'task': '3c',
    #'max_per_class': FRAMES_PER_VIDEO * 3,
    # RESULTS
    'base_results_dir': './results/',
    'n_plot': 20
}

add_config(configs, 'prednet_kitti__moments__prediction', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to evaluate predictions.',
             'model_name': 'prednet_kitti',
             'output_mode': 'prediction' }, eval_base_config)

add_config(configs, 'prednet_random__moments__representation', 
           { 'description': 'Using PredNet with random weights to extract features.',
             'model_name': 'prednet_random',
             'output_mode': 'representation',
           }, eval_base_config)

add_config(configs, 'prednet_kitti__moments__representation', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to extract features.',
             'model_name': 'prednet_kitti',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__representation', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_random_finetuned_moments__representation', 
           { 'description': 'Using PredNet trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_random_finetuned_moments',
             'output_mode': 'representation' }, eval_base_config)

train_base_config = dict()
train_base_config.update(eval_base_config)
train_base_config.update({
    'output_mode': 'error',
    'epochs': 150,
    'batch_size': SEQUENCES_PER_VIDEO,
    'shuffle': True,
    'stopping_patience': 20,
    'training_index_start': 0,
    'training_max_per_class': FRAMES_PER_VIDEO * 2,#400,
    'stack_sizes': (32, 64, 128, 256)
})

add_config(configs, 'prednet_kitti__moments__model', 
           { 'description': 'Training PredNet (pre-trained on KITTI) on Moments in Time dataset.',
             'model_name': 'prednet_kitti' }, train_base_config)

add_config(configs, 'prednet_random__moments__model', 
           { 'description': 'Training PredNet from scratch on Moments in Time dataset.',
             'model_name': 'prednet_random' }, train_base_config)    