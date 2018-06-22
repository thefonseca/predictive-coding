EMBEDDINGS_PER_VIDEO = 5

configs = {
    'moments_2c_transfer_kitti_R3': {
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
    }
    
}

