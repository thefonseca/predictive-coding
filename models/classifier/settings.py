import os

def add_config(configs, name, config, base_config=None):
    new_config = dict()
    if base_config:
        new_config.update(base_config)        
    new_config.update(config)
    new_config.update(tasks[new_config['task']])
    configs[name] = new_config

configs = {}

FRAMES_PER_VIDEO = 90

configs['imagenet__moments__features'] = {
    'description': 'Extract features from Moments in Time dataset',
    'input_shape': (160, 160, 3),
    'batch_size': 10,
    'sample_step': 3,
    'task': '10c',
    'model_type': 'vgg',
    'base_results_dir': './results',
    'training_data_dir': '../../datasets/moments_video_frames/training',
    'validation_data_dir': '../../datasets/moments_video_frames/validation'
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
    'dropout': 0.9,
    #'workers': 4,
    #'use_multiprocessing': True,
    'task': '2c_easy',
    'model_type': 'lstm',
    'hidden_dims': [64],
    'training_index_start': 0.6,
    'training_max_per_class': 0.2,
    'test_index_start': 0.8,
    'test_max_per_class': 0.2,
    'base_results_dir': './results',
    'training_data_dir': './results/vgg__imagenet__moments__features__10c/training',
    'validation_data_dir': './results/vgg__imagenet__moments__features__10c/validation',
    'test_data_dir': './results/vgg__imagenet__moments__features__10c/training',
}

VGG_SAMPLE_STEP = 2
FRAME_SAMPLE_STEP = 6

add_config(configs, 'moments__vgg_imagenet', 
           { 'description': 'A ConvLSTM classifier using VGG features',
             'seq_length': VGG_FEATURES_PER_VIDEO/VGG_SAMPLE_STEP,
             'min_seq_length': VGG_FEATURES_PER_VIDEO/VGG_SAMPLE_STEP,
             'sample_step': VGG_SAMPLE_STEP }, vgg_base_config)

add_config(configs, 'moments__images', 
           { 'description': 'A ConvLSTM classifier using raw images',
             'seq_length': FRAMES_PER_VIDEO/FRAME_SAMPLE_STEP,
             'sample_step': FRAME_SAMPLE_STEP,
             'input_channels': 3, 
             'input_height': 128, 
             'input_width': 160,
             'rescale': 1./255,
             'pad_sequences': True,
             'average_predictions': True,
             'training_data_dir': '../../datasets/moments_video_frames/training',
             'validation_data_dir': '../../datasets/moments_video_frames/validation',
             'test_data_dir': '../../datasets/moments_video_frames/training' }, vgg_base_config)

PREDNET_FEATURES_PER_VIDEO = 5

prednet_base_config = dict()
prednet_base_config.update(vgg_base_config)
prednet_base_config.update({
    'seq_length': PREDNET_FEATURES_PER_VIDEO,
    'min_seq_length': PREDNET_FEATURES_PER_VIDEO,
    'training_index_start': 0, #PREDNET_FEATURES_PER_VIDEO * 300,
    'training_max_per_class': 0.5,
    'test_index_start': 0.5,
    'test_max_per_class': None, # features_per_video * max_videos_per_class
    'hidden_dims': [64],
    'model_type': 'lstm'
})

add_config(configs, 'prednet_kitti_moments', 
           { 'description': 'A convnet classifier trained on PredNet \
(pretrained on KITTI) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_kitti__moments__representation__10c/training',
             'validation_data_dir': '../prednet/results/prednet_kitti__moments__representation__10c/validation',
             'test_data_dir': '../prednet/results/prednet_kitti__moments__representation__10c/training'}, prednet_base_config)

add_config(configs, 'prednet_random_moments', 
           { 'description': 'A convnet classifier trained on PredNet \
(random weights) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_random__moments__representation__10c/training',
             'validation_data_dir': '../prednet/results/prednet_random__moments__representation__10c/validation',
             'test_data_dir': '../prednet/results/prednet_random__moments__representation__10c/training'}, prednet_base_config)

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

add_config(configs, 'prednet_kitti_finetuned_moments_full', 
           { 'description': 'A convnet classifier trained on PredNet \
(pretrained on Moments in Time) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__full/training',
             'validation_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__full/validation',
             'test_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__representation__full/training'}, prednet_base_config)

add_config(configs, 'prednet_random_finetuned_moments_audio_10c',
           { 'description': 'A classifier trained on PredNet \
(pretrained on Moments in Time audio) features extracted from the Moments in Time dataset.',
             'training_data_dir': '../prednet/results/prednet_random_finetuned_moments_audio__representation__10c/training',
             'validation_data_dir': '../prednet/results/prednet_random_finetuned_moments_audio__representation__10c/validation',
             'test_data_dir': '../prednet/results/prednet_random_finetuned_moments_audio__representation__10c/training'}, prednet_base_config)

'''add_config(configs, 'vgg_prednet_ensemble', 
           { 'description': 'An ensemble classifier trained on features extracted from the Moments in Time dataset.',
             'task': '2c_easy',
             'model_type': 'convlstm',
             'batch_size': 10,
             'shuffle': False,
             'base_results_dir': './results',
             'ensemble': ['moments__vgg_imagenet', # 'moments__vgg_imagenet'] 
                          'prednet_kitti_finetuned_moments_10c']
           }, dict())'''

add_config(configs, 'prednet_kitti_finetuned_moments_full__ucf_01', 
           { 'description': 'A classifier trained on PredNet \
features extracted from the UCF-101 dataset.',
             'task': 'full',
             'seq_length': PREDNET_FEATURES_PER_VIDEO,
             'batch_size': 20,
             'min_seq_length': 1,
             'pad_sequences': True,
             'average_predictions': True,
             'training_max_per_class': .9,
             'training_index_start': 0,
             'validation_index_start': .9,
             'test_max_per_class': None,
             'test_index_start': 0,
             'training_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__ucf_01__representation__full/training',
             'validation_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__ucf_01__representation__full/training',
             'test_data_dir': '../prednet/results/prednet_kitti_finetuned_moments__ucf_01__representation__full/validation'}, prednet_base_config)

UCF_DATA_DIR = '../../datasets/ucf_data/'

add_config(configs, 'prednet_end2end_finetuned_moments_full__ucf_01', 
           { 'description': 'An end2end Prednet classifier trained on the UCF-101 dataset.',
             'task': 'full',
             'seq_length': 10,
             'min_seq_length': 5,
             'sample_step': 3,
             'input_channels': 3, 
             'input_height': 128, 
             'input_width': 160,
             'data_format': 'channels_first',
             'rescale': 1./255,
             'batch_size': 10,
             'pad_sequences': True,
             'model_type': 'pred_lstm',
             'hidden_dims': [32],
             'dropout': 0.5,
             'average_predictions': True,
             'training_max_per_class': .9,
             'training_index_start': 0,
             'validation_index_start': .9,
             'test_max_per_class': None,
             'test_index_start': 0,
             'training_data_dir': os.path.join(UCF_DATA_DIR, 'train_01'),
             'validation_data_dir': os.path.join(UCF_DATA_DIR, 'train_01'),
             'test_data_dir': os.path.join(UCF_DATA_DIR, 'test_01'),
             'model_weights_file': '../prednet/results/prednet_kitti__moments__model__full/weights.hdf5',
             'model_json_file': '../prednet/results/prednet_kitti__moments__model__full/model.json' }, prednet_base_config)