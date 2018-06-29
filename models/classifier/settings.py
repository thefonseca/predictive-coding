
def add_config(configs, name, config, base_config=None):
    new_config = dict()
    if base_config:
        new_config.update(base_config)        
    new_config.update(config)
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

VGG_FEATURES_PER_VIDEO = 30

convnet_base_config = {
    'epochs': 100,
    'stopping_patience': 10,
    'batch_size': 10,
    'shuffle': False,
    'dropout': 0.5,
    #'workers': 4,
    #'use_multiprocessing': True,
    'training_max_per_class': VGG_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
    'test_max_per_class': VGG_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
    'base_results_dir': './results',
    'training_data_dir': './results/vgg_imagenet__moments_nano__features/training',
    'validation_data_dir': './results/vgg_imagenet__moments_nano__features/validation',
    'test_data_dir': './results/vgg_imagenet__moments_nano__features/training',
}

add_config(configs, 'convnet__moments_nano__images_easy', 
           { 'description': 'A convnet classifier using Moments in Time images',
             'training_data_dir': '../../datasets/moments_nano_frames/training',
             'validation_data_dir': '../../datasets/moments_nano_frames/validation',
             'test_data_dir': '../../datasets/moments_nano_frames/training',
             'input_shape': (128, 160, 3),
             'sample_step': 3,
             'classes': ['cooking', 'walking']}, convnet_base_config)

add_config(configs, 'convnet__moments_nano__images_hard', 
           { 'description': 'A convnet classifier using Moments in Time images',
             'training_data_dir': '../../datasets/moments_nano_frames/training',
             'validation_data_dir': '../../datasets/moments_nano_frames/validation',
             'test_data_dir': '../../datasets/moments_nano_frames/training',
             'input_shape': (128, 160, 3),
             'sample_step': 3,
             'classes': ['running', 'walking']}, convnet_base_config)
    
add_config(configs, 'convlstm__moments_nano__vgg_imagenet_easy', 
           { 'description': 'A ConvLSTM classifier using VGG features',
             'seq_length': VGG_FEATURES_PER_VIDEO,
             'model_type': 'convlstm',
             'classes': ['cooking', 'walking']}, convnet_base_config)

add_config(configs, 'convlstm__moments_nano__vgg_imagenet_hard', 
           { 'description': 'A ConvLSTM classifier using VGG features',
             'seq_length': VGG_FEATURES_PER_VIDEO,
             'model_type': 'convlstm',
             'classes': ['running', 'walking']}, convnet_base_config)

add_config(configs, 'lstm__moments_nano__vgg_imagenet_easy', 
           { 'description': 'A ConvLSTM classifier using VGG features',
             'seq_length': VGG_FEATURES_PER_VIDEO,
             'model_type': 'lstm',
             'classes': ['cooking', 'walking']}, convnet_base_config)

add_config(configs, 'lstm__moments_nano__vgg_imagenet_hard', 
           { 'description': 'A ConvLSTM classifier using VGG features',
             'seq_length': VGG_FEATURES_PER_VIDEO,
             'model_type': 'lstm',
             'classes': ['running', 'walking']}, convnet_base_config)


PREDNET_FEATURES_PER_VIDEO = 5

prednet_base_config = dict()
prednet_base_config.update(convnet_base_config)
prednet_base_config.update({
    'seq_length': PREDNET_FEATURES_PER_VIDEO,
    'training_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class,
    'test_max_per_class': PREDNET_FEATURES_PER_VIDEO * 100, # features_per_video * max_videos_per_class
})

add_config(configs, 'convlstm__moments_nano__prednet_kitti_R3_easy', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on KITTI) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'convlstm',
             'training_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'classes': ['cooking', 'walking']}, prednet_base_config)

add_config(configs, 'convlstm__moments_nano__prednet_kitti_R3_hard', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on KITTI) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'convlstm',
             'training_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'classes': ['running', 'walking']}, prednet_base_config)

add_config(configs, 'convlstm__moments_nano__prednet_random_R3_easy', 
           { 'description': 'A convnet classifier trained on the PredNet \
(random weights) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'convlstm',
             'training_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_random__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'classes': ['cooking', 'walking']}, prednet_base_config)

add_config(configs, 'convlstm__moments_nano__prednet_random_R3_hard', 
           { 'description': 'A convnet classifier trained on the PredNet \
(random weights) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'convlstm',
             'training_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_random__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'classes': ['running', 'walking']}, prednet_base_config)

add_config(configs, 'convlstm__moments_nano__prednet_moments_v10_R3_easy', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on Moments in Time) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'convlstm',
             'training_data_dir': '../prednet/results/prednet_10v__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_10v__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_10v__moments_nano__R3/training',
             'classes': ['cooking', 'walking']}, prednet_base_config)

add_config(configs, 'convlstm__moments_nano__prednet_moments_v10_R3_hard', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on Moments in Time) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'convlstm',
             'training_data_dir': '../prednet/results/prednet_10v__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_10v__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_10v__moments_nano__R3/training',
             'classes': ['running', 'walking']}, prednet_base_config)

add_config(configs, 'lstm__moments_nano__prednet_kitti_R3_easy', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on KITTI) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'lstm',
             'training_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'classes': ['cooking', 'walking']}, prednet_base_config)

add_config(configs, 'lstm__moments_nano__prednet_kitti_R3_hard', 
           { 'description': 'A convnet classifier trained on the PredNet \
(pretrained on KITTI) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'lstm',
             'training_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments_nano__R3/training',
             'classes': ['running', 'walking']}, prednet_base_config)

add_config(configs, 'lstm__moments_nano__prednet_random_R3_easy', 
           { 'description': 'A convnet classifier trained on the PredNet \
(random weights) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'lstm',
             'training_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_random__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'classes': ['cooking', 'walking']}, prednet_base_config)

add_config(configs, 'lstm__moments_nano__prednet_random_R3_hard', 
           { 'description': 'A convnet classifier trained on the PredNet \
(random weights) R3 features extracted from the Moments in Time dataset.',
             'model_type': 'lstm',
             'training_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'validation_data_dir': '../prednet/results/prednet_random__moments_nano__R3/validation',
             'test_data_dir': '../prednet/results/prednet_random__moments_nano__R3/training',
             'classes': ['running', 'walking']}, prednet_base_config)