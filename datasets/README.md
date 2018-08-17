# Datasets

This folder contains scripts to download and process the datasets used in experiments.

## KITTI

The processed version of the [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset can be downloaded by running `kitti_download.sh`. For more information, see the original [PredNet repository](https://github.com/coxlab/prednet).

## Moments in Time

The [Moments in Time](http://moments.csail.mit.edu) dataset is a large collection of short videos "involving people, animals, objects or natural phenomena, that capture the gist of a dynamic scene". To download the smaller version, run `moments_download.sh`. This will download and extract the set of videos organized in folders for each activity class. 

To extract the frames for input in the PredNet model, run `moments_frames.sh`. This is generate a folder named `moments_data_frames` with images of frames organized in folders corresponding to each activity class. It will also generate pickle files with sources of each frame (used to generate batches without accessing the the disk).

## UCF-101
* Download and extract the dataset the dataset [here](http://www.thumos.info/download.html).
* Run `ucf_move_files.py` to move files according to train/test split
* Run `ucf_extract_frames.py` to extract frames from videos

