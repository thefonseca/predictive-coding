import os

# Where KITTI data will be saved if you run process_kitti.py
# If you directly download the processed data, change to the path of the data.
DATA_DIR = '../../datasets/moments_data_frames/'
DATA_DIR_TOY = '../../datasets/moments_toy_frames/'

experiments = {
    'moments_toy_transfer_kitti': {
        'description': 'Using PredNet pre-trained on KITTI dataset to perform inference \
on Moments in Time dataset.',
        'n_timesteps': 10,
        'frame_step': 3,
        'seq_overlap': 5,
        'max_seq_per_video': 5,
        'batch_size': 5,
        'max_missing_frames': 15,
        #'N_seq': 5,
        'shuffle': False,
        'seed': 17,
        #'gpus': 4,
        
        # WEIGHTS
        'model_weights_file': './model_data/kitti_keras/prednet_kitti_weights.hdf5',
        'model_json_file': './model_data/kitti_keras/prednet_kitti_model.json',
        
        # DATA
        'training_img_dir': os.path.join(DATA_DIR_TOY, 'training'),
        'training_img_sources': os.path.join(DATA_DIR_TOY, 'sources_training.pkl'),
        'validation_img_dir': os.path.join(DATA_DIR_TOY, 'validation'),
        'validation_img_sources': os.path.join(DATA_DIR_TOY, 'sources_validation.pkl'),
        'test_img_dir': os.path.join(DATA_DIR_TOY, 'test'),
        'test_img_sources': os.path.join(DATA_DIR_TOY, 'sources_test.pkl'),
        
        # RESULTS
        'base_results_dir': './results/',
        'n_plot': 20
    }
    
}
