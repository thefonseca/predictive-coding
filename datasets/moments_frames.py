import os
import glob
from tqdm import tqdm
import argparse

FLAGS = None

def extract_frames(source_dir, dest_dir, splits, video_pattern='{}/**/*.mp4'):
    
    video_pattern = os.path.join(source_dir, video_pattern)
    video_pattern2 = video_pattern.replace('/**', '')
    
    for split in splits:

        current_folder = None
        
        videos = glob.glob(video_pattern.format(split))
        
        if len(videos) == 0:
            videos = glob.glob(video_pattern2.format(split))

        for video in tqdm(videos, desc='Processing {} set'.format(split)):

            folder, filename = os.path.split(video)
            category = os.path.split(folder)[1]

            frame_folder = os.path.join(dest_dir, split)

            if split != category:
                frame_folder = os.path.join(frame_folder, category)

            frame_pattern = os.path.splitext(filename)[0] + '__frame_%03d.jpg'

            if not os.path.exists(frame_folder): 
                os.makedirs(frame_folder)
                current_folder = frame_folder
            elif current_folder != frame_folder:
                # if directory was not created in this run, skip to
                # avoid overwritting 
                continue

            frame_path = os.path.join(frame_folder, frame_pattern)
            #print('Extracting {} frames to {} ...'.format(video, frame_path))
            os.system('ffmpeg -hide_banner -loglevel panic -i "{}" "{}"'.format(video, frame_path))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from videos using ffmpeg.')
    parser.add_argument('source_dir', help='directory storing raw videos')
    parser.add_argument('dest_dir', help='directory for storing extracted frames')
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['training', 'validation', 'test'],
                        help='subdirs of "source_dir" containing videos for each dataset split')
    FLAGS, unparsed = parser.parse_known_args()
    
    extract_frames(FLAGS.source_dir, FLAGS.dest_dir, FLAGS.splits)
    
