import os

DATA_DIR = '../../datasets/moments_video_frames/'
AUDIO_DIR = '../../datasets/moments_audio_frames/'
UCF_DATA_DIR = '../../datasets/ucf_data/'
UCF_AUDIO_DIR = '../../datasets/ucf_audio/'
FRAMES_PER_VIDEO = 90
AUDIO_FRAMES_PER_VIDEO = 30
SEQUENCES_PER_VIDEO = 5

def add_config(configs, name, config, base_config=None):
    new_config = dict()
    if base_config:
        new_config.update(base_config)       
    new_config.update(config)
    new_config.update(tasks[new_config['task']])
    new_config.update(models[new_config['model_name']])
    configs[name] = new_config
    

configs = {}

tasks = {
    '3c': {
        'classes': ['cooking', 'running', 'walking']
    },
    '10c': {
        'classes': ['barking', 'cooking', 'driving', 'juggling', 'photographing', 
                    'biting', 'climbing', 'running', 'sleeping', 'walking']
    },
    'full': {
        'classes': None
    }
}

models = {
    'prednet_kitti': {
        'model_weights_file': './kitti/model_data/kitti_keras/prednet_kitti_weights.hdf5',
        'model_json_file': './kitti/model_data/kitti_keras/prednet_kitti_model.json',
    },
    'prednet_random': {
        'model_weights_file': None,
        'model_json_file': None
    },
    'prednet_kitti_finetuned_moments': {
        'model_weights_file': './results/prednet_kitti__moments__model{}/weights.hdf5',
        'model_json_file': './results/prednet_kitti__moments__model{}/model.json',
    },
    'prednet_random_finetuned_moments': {
        'model_weights_file': './results/prednet_random__moments__model{}/weights.hdf5',
        'model_json_file': './results/prednet_random__moments__model{}/model.json',
    },
    'prednet_random_finetuned_moments_audio': {
        'model_weights_file': './results/prednet_random__moments_audio__model{}/weights.hdf5',
        'model_json_file': './results/prednet_random__moments_audio__model{}/model.json',
    },
    'prednet_finetuned_ucf': {
        'model_weights_file': './results/prednet__ucf_01__model{}/weights.hdf5',
        'model_json_file': './results/prednet__ucf_01__model{}/model.json',
    }
}

eval_base_config = {
    'n_timesteps': 10,
    'min_seq_length': 10,
    'frame_step': 3,
    'seq_overlap': 5,
    'timestep_start': -1,
    'batch_size': 1,
    'stateful': False,
    'input_channels': 3, 
    'input_height': 128, 
    'input_width': 160,
    'rescale': 1./255,
    'shuffle': False,
    'workers': 4,
    # DATA
    'training_data_dir': os.path.join(DATA_DIR, 'training'),
    'validation_data_dir': os.path.join(DATA_DIR, 'validation'),
    'task': '10c',
    'pretrained': '10c',
    # extract features only for the last 40% videos
    'training_index_start': 0.6,
    'training_max_per_class': 0.4,
    'test_index_start': 0.8,
    'test_max_per_class': 0.2,
    # RESULTS
    'base_results_dir': './results/',
    'n_plot': 20
}

add_config(configs, 'prednet_random__moments__prediction', 
           { 'description': 'Using PredNet with random weights to evaluate predictions.',
             'model_name': 'prednet_random',
             'output_mode': 'prediction' }, eval_base_config)

add_config(configs, 'prednet_kitti__moments__prediction', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to evaluate predictions.',
             'model_name': 'prednet_kitti',
             'output_mode': 'prediction' }, eval_base_config)

add_config(configs, 'prednet_random__moments_audio__prediction', 
           { 'description': 'Training PredNet from scratch on Moments in Time dataset audio spectrograms.',
             'frame_step': 1,
             'training_data_dir': os.path.join(AUDIO_DIR, 'training'),
             'validation_data_dir': os.path.join(AUDIO_DIR, 'validation'),
             'model_name': 'prednet_random' }, eval_base_config)

add_config(configs, 'prednet_random__moments__representation', 
           { 'description': 'Using PredNet with random weights to extract features.',
             'model_name': 'prednet_random',
             'output_mode': 'representation',
           }, eval_base_config)

add_config(configs, 'prednet_kitti__moments__representation', 
           { 'description': 'Using PredNet pre-trained on KITTI dataset to extract features.',
             'model_name': 'prednet_kitti',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__prediction', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'output_mode': 'prediction' }, eval_base_config)

add_config(configs, 'prednet_random_finetuned_moments__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_random_finetuned_moments',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_random_finetuned_moments_audio__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time audio dataset to extract features.',
             'model_name': 'prednet_random_finetuned_moments_audio',
             'frame_step': 1,
             'training_data_dir': os.path.join(AUDIO_DIR, 'training'),
             'validation_data_dir': os.path.join(AUDIO_DIR, 'validation'),
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_random__moments_audio__representation', 
           { 'description': 'Using PredNet with random weights to extract features.',
             'model_name': 'prednet_random',
             'frame_step': 1,
             'training_data_dir': os.path.join(AUDIO_DIR, 'training'),
             'validation_data_dir': os.path.join(AUDIO_DIR, 'validation'),
             'output_mode': 'representation'
           }, eval_base_config)


add_config(configs, 'prednet_random__ucf_01__representation', 
           { 'description': 'Using PredNet (random weights) to extract video features.',
             'model_name': 'prednet_random',
             'training_data_dir': os.path.join(UCF_DATA_DIR, 'train_01'),
             'validation_data_dir': os.path.join(UCF_DATA_DIR, 'test_01'),
             'task': 'full',
             'min_seq_length': 5,
             'pad_sequences': True,
             'pretrained': 'full',
             'training_index_start': 0,
             'training_max_per_class': None,
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__ucf_01__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'training_data_dir': os.path.join(UCF_DATA_DIR, 'train_01'),
             'validation_data_dir': os.path.join(UCF_DATA_DIR, 'test_01'),
             'task': 'full',
             'min_seq_length': 5,
             'pad_sequences': True,
             'pretrained': 'full',
             'training_index_start': 0,
             'training_max_per_class': None,
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_random__ucf_01_audio__representation', 
           { 'description': 'Using PredNet (random weights) to extract audio features.',
             'model_name': 'prednet_random_finetuned_moments_audio',
             'training_data_dir': os.path.join(UCF_AUDIO_DIR, 'train'),
             'validation_data_dir': os.path.join(UCF_AUDIO_DIR, 'test'),
             'task': 'full',
             'frame_step': 1,
             'min_seq_length': 5,
             'pad_sequences': True,
             'pretrained': 'full',
             'training_index_start': 0,
             'training_max_per_class': None,
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_random_finetuned_moments_audio__ucf_01__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_random_finetuned_moments_audio',
             'training_data_dir': os.path.join(UCF_AUDIO_DIR, 'train'),
             'validation_data_dir': os.path.join(UCF_AUDIO_DIR, 'test'),
             'task': 'full',
             'frame_step': 1,
             'min_seq_length': 5,
             'pad_sequences': True,
             'pretrained': 'full',
             'training_index_start': 0,
             'training_max_per_class': None,
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_finetuned_ucf__ucf_01__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_finetuned_ucf',
             'training_data_dir': os.path.join(UCF_DATA_DIR, 'train_01'),
             'validation_data_dir': os.path.join(UCF_DATA_DIR, 'test_01'),
             'task': 'full',
             'min_seq_length': 5,
             'pad_sequences': True,
             'pretrained': 'full',
             'training_index_start': 0,
             'training_max_per_class': None,
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__ucf_02__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'training_data_dir': os.path.join(UCF_DATA_DIR, 'train_02'),
             'validation_data_dir': os.path.join(UCF_DATA_DIR, 'test_02'),
             'task': 'full',
             'min_seq_length': 5,
             'pad_sequences': True,
             'pretrained': 'full',
             'training_index_start': 0,
             'training_max_per_class': None,
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__ucf_03__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'training_data_dir': os.path.join(UCF_DATA_DIR, 'train_03'),
             'validation_data_dir': os.path.join(UCF_DATA_DIR, 'test_03'),
             'task': 'full',
             'min_seq_length': 5,
             'pad_sequences': True,
             'pretrained': 'full',
             'training_index_start': 0,
             'training_max_per_class': None,
             'output_mode': 'representation' }, eval_base_config)

train_base_config = dict()
train_base_config.update(eval_base_config)
train_base_config.update({
    'output_mode': 'error',
    'epochs': 50,
    'batch_size': 2 * SEQUENCES_PER_VIDEO,
    'shuffle': True,
    'task': '10c',
    #'gpus': 2,
    #'stopping_patience': 100,
    'training_index_start': 0,
    # train on first 80% videos
    'training_max_per_class': 0.8, #AUDIO_FRAMES_PER_VIDEO * 400,
    #'stack_sizes': (32, 64, 128, 256)
})

add_config(configs, 'prednet_kitti__moments__model', 
           { 'description': 'Training PredNet (pre-trained on KITTI) on Moments in Time dataset.',
             'model_name': 'prednet_kitti' }, train_base_config)

add_config(configs, 'prednet_random__moments__model', 
           { 'description': 'Training PredNet from scratch on Moments in Time dataset.',
             'model_name': 'prednet_random' }, train_base_config)

add_config(configs, 'prednet__ucf_01__model', 
           { 'description': 'Training PredNet on UCF-101 (split 1) dataset.',
             'training_data_dir': os.path.join(UCF_DATA_DIR, 'train_01'),
             'validation_data_dir': os.path.join(UCF_DATA_DIR, 'train_01'),
             'task': 'full',
             'pretrained': 'full',
             'training_max_per_class': 0.9,
             'validation_index_start': 0.9,
             'model_name': 'prednet_kitti_finetuned_moments' }, train_base_config)

add_config(configs, 'prednet_random__moments_audio__model', 
           { 'description': 'Training PredNet from scratch on Moments in Time dataset audio spectrograms.',
             'frame_step': 1,
             'training_data_dir': os.path.join(AUDIO_DIR, 'training'),
             'validation_data_dir': os.path.join(AUDIO_DIR, 'validation'),
             'model_name': 'prednet_random' }, train_base_config)
