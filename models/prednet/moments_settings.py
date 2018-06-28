import os

DATA_DIR = '../../datasets/moments_nano_frames/'

def add_config(configs, name, config, base_config=None):
    new_config = dict()
    if base_config:
        new_config.update(base_config)       
    new_config.update(config)
    configs[name] = new_config

configs = {}

base_config = {  
    'n_timesteps': 10,
    'frame_step': 3,
    'seq_overlap': 5,
    'timestep_start': -1,
    'max_seq_per_video': 5,
    'batch_size': 20,
    'max_missing_frames': 15,
    'input_shape': (128,160,3),
    'n_channels': 3, 
    'img_height': 128, 
    'img_width': 160,
    #'N_seq': 5,
    'shuffle': False,
    #'workers': 4,
    #'use_multiprocessing': True,
    # DATA
    'training_data_dir': os.path.join(DATA_DIR, 'training'),
    'training_data_sources': os.path.join(DATA_DIR, 'sources_training.pkl'),
    'validation_data_dir': os.path.join(DATA_DIR, 'validation'),
    'validation_data_sources': os.path.join(DATA_DIR, 'sources_validation.pkl'),
    'model_weights_file': './model_data/kitti_keras/prednet_kitti_weights.hdf5',
    'model_json_file': './model_data/kitti_keras/prednet_kitti_model.json',
    # RESULTS
    'base_results_dir': './results/',
    'n_plot': 20
}

add_config(configs, 'prednet_kitti__moments_nano__prediction', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to evaluate predictions.',
             'output_mode': 'prediction' }, base_config)

add_config(configs, 'prednet_random__moments_nano__R3', 
           { 'description': 'Using PredNet with random weights to extract R3 features.',
             'output_mode': 'R3',
             'model_weights_file': None,
             'model_json_file': None }, base_config)

add_config(configs, 'prednet_kitti__moments_nano__R3', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to extract R3 features.',
             'output_mode': 'R3' }, base_config)

FRAMES_PER_VIDEO = 90

add_config(configs, 'prednet__moments_nano__train_10v', 
           { 'description': 'Training PredNet on Moments in Time dataset.',
             'output_mode': 'error',
             'epochs': 150,
             'batch_size': 4,
             # We start at video #250 to avoid using the same videos
             # present in the classifier (convnet) dataset
             'training_index_start': FRAMES_PER_VIDEO * 250,
             'training_max_per_class': FRAMES_PER_VIDEO * 10 }, base_config)

add_config(configs, 'prednet__moments_nano__train_50v', 
           { 'description': 'Training PredNet on Moments in Time dataset.',
             'output_mode': 'error',
             'epochs': 150,
             'batch_size': 4,
             # We start at video #250 to avoid using the same videos
             # present in the classifier (convnet) dataset
             'training_index_start': FRAMES_PER_VIDEO * 250,
             'training_max_per_class': FRAMES_PER_VIDEO * 50 }, base_config)

add_config(configs, 'prednet__moments_nano__train_100v', 
           { 'description': 'Training PredNet on Moments in Time dataset.',
             'output_mode': 'error',
             'epochs': 150,
             'batch_size': 4,
             # We start at video #250 to avoid using the same videos
             # present in the classifier (convnet) dataset
             'training_index_start': FRAMES_PER_VIDEO * 250,
             'training_max_per_class': FRAMES_PER_VIDEO * 100 }, base_config)
    
