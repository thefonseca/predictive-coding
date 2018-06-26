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
    #'frames_per_video': 90,
    #'max_videos_per_class': 200,
    #'N_seq': 5,
    'shuffle': False,
    #'workers': 4,
    #'use_multiprocessing': True,
    # DATA
    'training_img_dir': os.path.join(DATA_DIR, 'training'),
    'training_img_sources': os.path.join(DATA_DIR, 'sources_training.pkl'),
    'validation_img_dir': os.path.join(DATA_DIR, 'validation'),
    'validation_img_sources': os.path.join(DATA_DIR, 'sources_validation.pkl'),
    #'test_img_dir': os.path.join(DATA_DIR, 'test'),
    #'test_img_sources': os.path.join(DATA_DIR, 'sources_test.pkl'),
    # RESULTS
    'base_results_dir': './results/',
    'n_plot': 20
}

add_config(configs, 'prednet_kitti__moments_nano__prediction', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to evaluate predictions.',
             'output_mode': 'prediction',
             'model_weights_file': './model_data/kitti_keras/prednet_kitti_weights.hdf5',
             'model_json_file': './model_data/kitti_keras/prednet_kitti_model.json'}, base_config)

add_config(configs, 'prednet_kitti__moments_nano__R3', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to extract R3 features.',
             'output_mode': 'R3',
             'model_weights_file': './model_data/kitti_keras/prednet_kitti_weights.hdf5',
             'model_json_file': './model_data/kitti_keras/prednet_kitti_model.json'}, base_config)
    
