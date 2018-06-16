import os

# Where KITTI data will be saved if you run process_kitti.py
# If you directly download the processed data, change to the path of the data.
DATA_DIR = '../datasets/moments_data_frames/'

experiments = {
    'moments_transfer_kitti': {
        'description': 'Using PredNet pre-trained on KITTI dataset to perform inference \
on Moments in Time dataset.',
        'n_timesteps': 10,
        'frame_step': 3,
        'seq_overlap': 5,
        'max_seq_per_video': 5,
        'batch_size': 5,
        'max_missing_frames': 15,
        'N_seq': 5,
        'shuffle': False,
        'seed': 17,
        #'gpus': 4,
        
        # WEIGHTS
        'model_weights_file': './model_data/kitti_keras/prednet_kitti_weights.hdf5',
        'model_json_file': './model_data/kitti_keras/prednet_kitti_model.json',
        
        # DATA
        'train_img_dir': os.path.join(DATA_DIR, 'training'),
        'train_img_sources': os.path.join(DATA_DIR, 'sources_training.pkl'),
        'validation_img_dir': os.path.join(DATA_DIR, 'validation'),
        'validation_img_sources': os.path.join(DATA_DIR, 'sources_validation.pkl'),
        'test_img_dir': os.path.join(DATA_DIR, 'test'),
        'test_img_sources': os.path.join(DATA_DIR, 'sources_test.pkl'),
        
        # RESULTS
        'base_results_dir': './results/',
        'n_plot': 20
    }
    
}
