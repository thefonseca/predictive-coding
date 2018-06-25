EMBEDDINGS_PER_VIDEO = 5

configs = {
    'prednet_kitti__moments_2c__R3': {
        'description': 'A convnet classifier trained on the R3 PredNet model \
(pretrained on KITTI) features extracted from the Moments in Time dataset (binary version).',
        'epochs': 5,
        'batch_size': 20,
        'shuffle': True,
        'seed': 17,
        'workers': 8,
        #'use_multiprocessing': True,
        #'gpus': 4,
        
        # WEIGHTS
        #'model_weights_file': './model_data/kitti_keras/prednet_kitti_weights.hdf5',
        #'model_json_file': './model_data/kitti_keras/prednet_kitti_model.json',
        
        # DATA
        'training_data_dir': '../prednet/results/moments_2c_transfer_kitti_R3/training',
        'validation_data_dir': '../prednet/results/moments_2c_transfer_kitti_R3/validation',
        'test_data_dir': '../prednet/results/moments_2c_transfer_kitti_R3/training',
        'training_max_per_class': 400 * EMBEDDINGS_PER_VIDEO,
        
        # RESULTS
        'base_results_dir': './results/'
    },
    
    'vgg_moments_2c' : {
        'description': 'Extract features from Moments in Time dataset',
        'input_shape': (160, 160, 3),
        'batch_size': 10,
        'frames_per_video': 90,
        'max_videos_per_class': 150,
        'sample_step': 3,
        'base_results_dir': './results',
        'training_data_dir': '../../datasets/moments_2c_frames/training',
        'validation_data_dir': '../../datasets/moments_2c_frames/validation'
    },
    
    'moments_2c__vgg_features' : {
        'description': 'A convnet classifier using VGG features',
        'epochs': 5,
        'batch_size': 20,
        'seq_length': 20,
        'shuffle': True,
        'dropout': 0.5,
        'seed': 17,
        'workers': 8,
        'use_multiprocessing': True,
        'training_max_per_class': 30 * 100, # frames_per_video * max_videos_per_class
        'base_results_dir': './results',
        'training_data_dir': './results/vgg__moments_2c__features/training',
        'validation_data_dir': './results/vgg__moments_2c__features/validation',
        'test_data_dir': './results/vgg__moments_2c__features/training'
    }
    
}

