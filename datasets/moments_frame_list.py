import os
import glob
from tqdm import tqdm
import pickle as pkl
import argparse

FLAGS = None

def generate_frame_list(source_dir, splits):
    image_pattern = os.path.join(source_dir, '{}/*/*.jpg')
    image_pattern2 = os.path.join(source_dir, '{}/*.jpg')

    for split in splits:

        source_list = []
        print('Searching for image files in "{}/{}/<category>" (this may take a while)...'.format(source_dir, 
                                                                                                  split))
        frames = sorted(glob.glob(image_pattern.format(split)))

        if len(frames) == 0:
            print('Searching for image files in "{}/{}" (this may take a while)...'.format(source_dir, 
                                                                                           split))
            frames = sorted(glob.glob(image_pattern2.format(split)))

        if len(frames) == 0:
            continue

        for frame in tqdm(frames, desc='Processing {} set'.format(split)):

            folder, filename = os.path.split(frame)
            category = os.path.split(folder)[1]
            source_video_name = filename.split('__frame_')[0]

            if split != category:
                source_list.append('{}__{}'.format(category, source_video_name))
            else:
                source_list.append(source_video_name)

        filename = os.path.join(source_dir, 'sources_' + split + '.pkl')
        pkl.dump(source_list, open(filename, "wb"))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from videos using ffmpeg.')
    parser.add_argument('source_dir', help='directory storing frame images')
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['training', 'validation', 'test'],
                        help='subdirs of "source_dir" containing videos for each dataset split')
    FLAGS, unparsed = parser.parse_known_args()
    
    print('Generating frame list for images in directory {}'.format(FLAGS.source_dir))
    generate_frame_list(FLAGS.source_dir, FLAGS.splits)
