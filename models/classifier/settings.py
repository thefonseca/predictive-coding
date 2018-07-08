
def add_config(configs, name, config, base_config=None):
    new_config = dict()
    if base_config:
        new_config.update(base_config)        
    new_config.update(config)
    new_config.update(tasks[new_config['task']])
    configs[name] = new_config

configs = {}

configs['vgg_imagenet__moments_nano__features'] = {
    'description': 'Extract features from Moments in Time dataset',
    'input_shape': (160, 160, 3),
    'batch_size': 10,
    'frames_per_video': 90,
    'max_videos_per_class': 200,
    'sample_step': 3,
    'base_results_dir': './results',
    'training_data_dir': '../../datasets/moments_nano_frames/training',
    'validation_data_dir': '../../datasets/moments_nano_frames/validation'
}

tasks = {
    '2c_easy': {
        'classes': ['cooking', 'walking']
    },
    '2c_hard': {
        'classes': ['running', 'walking']
    },
    '10c': {
        'classes': ['barking', 'cooking', 'driving', 'juggling', 'photographing', 
                    'biting', 'climbing', 'running', 'sleeping', 'walking']
    },
    'full': {
        'classes': None
    }
}

VGG_FEATURES_PER_VIDEO = 30

vgg_base_config = {
    'epochs': 100,
    'stopping_patience': 50,
    'batch_size': 10,
    'shuffle': True,
    'dropout': 0.5,
    #'workers': 4,
    #'use_multiprocessing': True,
    'task': '2c_easy',
    #'training_index_start': VGG_FEATURES_PER_VIDEO * 300,
    'training_max_per_class': VGG_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
    'test_index_start': VGG_FEATURES_PER_VIDEO * 100,
    'test_max_per_class': VGG_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
    'base_results_dir': './results',
    'training_data_dir': './results/vgg_imagenet__moments_nano__features/training',
    'validation_data_dir': './results/vgg_imagenet__moments_nano__features/validation',
    'test_data_dir': './results/vgg_imagenet__moments_nano__features/training',
}

add_config(configs, 'moments__vgg_imagenet', 
           { 'description': 'A ConvLSTM classifier using VGG features',
             'seq_length': VGG_FEATURES_PER_VIDEO,
             'model_type': 'convlstm' }, vgg_base_config)

PREDNET_FEATURES_PER_VIDEO = 5

prednet_base_config = dict()
prednet_base_config.update(vgg_base_config)
prednet_base_config.update({
    'seq_length': PREDNET_FEATURES_PER_VIDEO,
    #'training_index_start': PREDNET_FEATURES_PER_VIDEO * 300,
    'training_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class,
    'test_index_start': PREDNET_FEATURES_PER_VIDEO * 100,
    'test_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
    'model_type': 'convlstm'
})

add_config(configs, 'prednet_kitti_moments', 
           { 'description': 'A convnet classifier trained on PredNet \
(pretrained on KITTI) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_kitti__moments__representation/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments__representation/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments__representation/training'}, prednet_base_config)

add_config(configs, 'prednet_random_moments', 
           { 'description': 'A convnet classifier trained on PredNet \
(random weights) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_random__moments__representation/training',
             'validation_data_dir': '../prednet/results/prednet_random__moments__representation/validation',
             'test_data_dir': '../prednet/results/prednet_random__moments__representation/training'}, prednet_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments_3c', 
           { 'description': 'A convnet classifier trained on PredNet \
(pretrained on Moments in Time) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__3c/training',
             'validation_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__3c/validation',
             'test_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__3c/training'}, prednet_base_config)

add_config(configs, 'prednet_random_finetuned_moments_3c',
           { 'description': 'A convnet classifier trained on PredNet \
(pretrained on Moments in Time) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_random_finetuned_moments__representation__3c/training',
             'validation_data_dir': '../prednet/results/prednet_random_finetuned_moments__representation__3c/validation',
             'test_data_dir': '../prednet/results/prednet_random_finetuned_moments__representation__3c/training'}, prednet_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments_10c', 
           { 'description': 'A convnet classifier trained on PredNet \
(pretrained on Moments in Time) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__10c/training',
             'validation_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__10c/validation',
             'test_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__10c/training'}, prednet_base_config)

add_config(configs, 'vgg_prednet_ensemble', 
           { 'description': 'An ensemble classifier trained on features extracted from the Moments in Time dataset.',
             'task': '2c_easy',
             'model_type': 'convlstm',
             'batch_size': 10,
             'shuffle': False,
             'base_results_dir': './results',
             'ensemble': ['moments_nano__vgg_imagenet', 
                          'prednet_kitti_finetuned_moments_10c']}, dict())

add_config(configs, 'prednet_kitti_finetuned_moments_10c__ucf', 
           { 'description': 'A convnet classifier trained on PredNet \
(pretrained on Moments in Time) features extracted from the Moments in Time dataset.',
             'task': 'full',
             'model_type': 'convnet',
             'seq_length': None,
             'batch_size': 50,
             'training_max_per_class': .9,
             'training_index_start': 0,
             'validation_index_start': .9,
             'test_max_per_class': None,
             'test_index_start': 0,
             'training_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__ucf__representation__full/training',
             'validation_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__ucf__representation__full/training',
             'test_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__ucf__representation__full/validation'}, prednet_base_config)