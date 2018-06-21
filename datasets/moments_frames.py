import os
import glob
from tqdm import tqdm
import argparse

FLAGS = None

def extract_frames(source_dir, dest_dir, splits, categories=None, 
                   max_per_category=None, video_pattern='{}/**/*.mp4'):
    
    pattern_all_categories = os.path.join(source_dir, video_pattern)
    pattern_no_categories = pattern_all_categories.replace('/**', '') # '{}/*.mp4'
    pattern_category = pattern_no_categories.replace('{}', '{}/{}') # '{}/{}/*.mp4'
    category_count = {}
    
    for split in splits:
        
        current_folder = None
        videos = sorted(glob.glob(pattern_no_categories.format(split)))
        
        if len(videos) == 0:
            
            if categories is None:
                all_categories = os.walk(os.path.join(source_dir, split)).next()[1]
                categories = all_categories
                    
            for c in categories:
                cat_videos = sorted(glob.glob(pattern_category.format(split, c)))[:max_per_category]
                videos.extend(cat_videos)

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
    parser.add_argument('--categories', type=str, nargs='+', 
                        help='a subset of categories to be processed. Default is all categories.')
    parser.add_argument('--max_per_category', type=int,
                        help='maximum number of videos per category. Default is all videos.')
    FLAGS, unparsed = parser.parse_known_args()
    
    extract_frames(FLAGS.source_dir, FLAGS.dest_dir, 
                   FLAGS.splits,  FLAGS.categories, FLAGS.max_per_category)
    
