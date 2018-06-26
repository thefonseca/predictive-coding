
def add_config(configs, name, config, base_config=None):
    new_config = dict()
    if base_config:
        new_config.update(base_config)
        
    new_config.update(config)
    configs[name] = new_config

configs = {}

configs['vgg__moments_nano__features'] = {
    'description': 'Extract features from Moments in Time dataset',
    'input_shape': (160, 160, 3),
    'batch_size': 10,
    'frames_per_video': 90,
    'max_videos_per_class': 200,
    'sample_step': 3,
    'base_results_dir': './results',
    'training_data_dir': '../../datasets/moments_nano_frames/training',
    'validation_data_dir': '../../datasets/moments_nano_frames/validation',
}

VGG_FEATURES_PER_VIDEO = 30

convnet_base_config = {
    'epochs': 100,
    'stopping_patience': 10,
    'batch_size': 10,
    'shuffle': True,
    'dropout': 0.5,
    'workers': 4,
    #'use_multiprocessing': True,
    'training_max_per_class': VGG_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
    'base_results_dir': './results',
    'training_data_dir': './results/vgg__moments_nano__features/training',
    'validation_data_dir': './results/vgg__moments_nano__features/validation',
    'test_data_dir': './results/vgg__moments_nano__features/training',
}
    
add_config(configs, 'convnet__moments_nano__vgg_features_easy', 
           { 'description': 'A convnet classifier using VGG features',
             'classes': ['cooking', 'walking']}, convnet_base_config)

add_config(configs, 'convnet__moments_nano__vgg_features_hard', 
           { 'description': 'A convnet classifier using VGG features',
             'classes': ['running', 'walking']}, convnet_base_config)

add_config(configs, 'convlstm__moments_nano__vgg_features_easy', 
           { 'description': 'A ConvLSTM classifier using VGG features',
             'seq_length': VGG_FEATURES_PER_VIDEO,
             'classes': ['cooking', 'walking']}, convnet_base_config)

add_config(configs, 'convlstm__moments_nano__vgg_features_hard', 
           { 'description': 'A ConvLSTM classifier using VGG features',
             'seq_length': VGG_FEATURES_PER_VIDEO,
             'classes': ['running', 'walking']}, convnet_base_config)

PREDNET_FEATURES_PER_VIDEO = 5

add_config(configs, 'convnet__moments_nano__prednet_kitti_R3_easy', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on KITTI) R3 features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'training_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'test_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'classes': ['cooking', 'walking']}, convnet_base_config)

add_config(configs, 'convnet__moments_nano__prednet_random_R3_easy', 
           { 'description': 'A convnet classifier trained on the PredNet \
(random weights) R3 features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_random__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'training_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'test_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'classes': ['cooking', 'walking']}, convnet_base_config)

add_config(configs, 'convnet__moments_nano__prednet_kitti_R3_hard', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on KITTI) R3 features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'training_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'test_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'classes': ['running', 'walking']}, convnet_base_config)

add_config(configs, 'convnet__moments_nano__prednet_random_R3_hard', 
           { 'description': 'A convnet classifier trained on the PredNet \
(random weights) R3 features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_random__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'training_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'test_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'classes': ['running', 'walking']}, convnet_base_config)

add_config(configs, 'convlstm__moments_nano__prednet_kitti_R3_easy', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on KITTI) R3 features extracted from the Moments in Time dataset.',
             'seq_length': PREDNET_FEATURES_PER_VIDEO,
             'training_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'training_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'test_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
             'classes': ['cooking', 'walking']}, convnet_base_config)