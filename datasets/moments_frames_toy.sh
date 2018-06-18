python moments_frames.py ./moments_data ./moments_toy_frames --splits training validation \
--categories running walking --max_per_category 10

python moments_frame_list.py ./moments_toy_frames
