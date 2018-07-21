import os
import glob
from tqdm import tqdm
import pickle as pkl
import argparse
import subprocess

FLAGS = None


def extract_frames(source_dir, dest_dir, splits, categories=None, 
                   max_per_category=None, audio=False, size='160x128',
                   video_pattern='{}/**/*.mp4'):
    
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
                    
            for c in sorted(categories):
                cat_videos = sorted(glob.glob(pattern_category.format(split, c)))[:max_per_category]
                videos.extend(cat_videos)

        for video in tqdm(videos, desc='Processing {} set'.format(split)):

            folder, filename = os.path.split(video)
            category = os.path.split(folder)[1]

            frame_folder = os.path.join(dest_dir, split)

            if split != category:
                frame_folder = os.path.join(frame_folder, category)
                
            if not os.path.exists(frame_folder): 
                os.makedirs(frame_folder)
                current_folder = frame_folder
            elif current_folder != frame_folder:
                # if directory was not created in this run, skip to
                # avoid overwritting
                continue

            max_filename_length = 200
            if audio:
                # create spectrogram video
                audio_pattern = os.path.splitext(filename)[0][:max_filename_length] + '__audio.mp4'
                audio_path = os.path.join(frame_folder, audio_pattern)
                ffmpeg_cmd = 'ffmpeg -hide_banner -loglevel panic -i {} -filter_complex '
                #ffmpeg_cmd += '"[0:a]showspectrum=s={}:mode=combined:slide=fullframe:'
                ffmpeg_cmd += '"[0:a]showspectrum=s={}:mode=combined:slide=scroll:'
                # overlap=0.895 results in 10 fps videos
                #ffmpeg_cmd += 'scale=log:color=intensity:overlap=0.895[v]" -map "[v]" -map 0:a {}'
                ffmpeg_cmd += 'scale=log:color=intensity:overlap=0[v]" -map "[v]" -map 0:a {}'
                os.system(ffmpeg_cmd.format(video, size, audio_path))
                
                if os.path.exists(audio_path):
                    # force 10fps video
                    audio_pattern_10fps = os.path.splitext(filename)[0][:max_filename_length] + '__10fps.mp4'
                    audio_path_10fps = os.path.join(frame_folder, audio_pattern_10fps)
                    ffmpeg_cmd = 'ffmpeg -hide_banner -loglevel panic -y -i {} -r 10 {}'
                    os.system(ffmpeg_cmd.format(audio_path, audio_path_10fps))
                    os.remove(audio_path)
                    audio_path = audio_path_10fps
                    
                '''if not os.path.exists(audio_path):
                    # If original video does not have audio, we create "silence" frames
                    duration_cmd = 'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}'
                    duration = float(subprocess.check_output(duration_cmd.format(video), shell=True))
                    #print('Creating "silence" video with {} seconds'.format(duration))
                    fps = 10
                    create_cmd = 'ffmpeg -hide_banner -loglevel panic -t {} -s {} -f rawvideo -pix_fmt rgb24 -r {} -i /dev/zero {}'
                    os.system(create_cmd.format(duration, size, fps, audio_path))'''
                    
                video = audio_path
            
            frame_pattern = os.path.splitext(filename)[0][:max_filename_length] + '__frame_%03d.jpg'
            frame_path = os.path.join(frame_folder, frame_pattern)
            #print('Extracting {} frames to {} ...'.format(video, frame_path))
            os.system('ffmpeg -hide_banner -loglevel panic -i "{}" "{}"'.format(video, frame_path))
            
            if audio and os.path.exists(video):
                os.remove(video)
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from videos using ffmpeg.')
    parser.add_argument('source_dir', help='directory storing raw videos')
    parser.add_argument('dest_dir', help='directory for storing extracted frames')
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['training', 'validation'],
                        help='subdirs of "source_dir" containing videos for each dataset split')
    parser.add_argument('--categories', type=str, nargs='+', 
                        help='a subset of categories to be processed. Default is all categories.')
    parser.add_argument('--max_per_category', type=int,
                        help='maximum number of videos per category. Default is all videos.')
    parser.add_argument('--audio', help='extract audio spectrograms.',
                        action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    
    extract_frames(FLAGS.source_dir, FLAGS.dest_dir, FLAGS.splits, 
                   FLAGS.categories, FLAGS.max_per_category, FLAGS.audio)
