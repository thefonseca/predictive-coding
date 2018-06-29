# Train all model variants

mkdir -p logs
# 2>&1 | tee "./logs/moments_2c_transfer_kitti_R3_$(date +"%FT%H%M").log"
python train.py --config convlstm__moments_nano__vgg_imagenet_easy
python train.py --config convlstm__moments_nano__vgg_imagenet_hard
python train.py --config convlstm__moments_nano__prednet_random_R3_easy
python train.py --config convlstm__moments_nano__prednet_random_R3_hard
python train.py --config convlstm__moments_nano__prednet_kitti_R3_easy
python train.py --config convlstm__moments_nano__prednet_kitti_R3_hard
python train.py --config convlstm__moments_nano__prednet_moments_v10_R3_easy
python train.py --config convlstm__moments_nano__prednet_moments_v10_R3_hard
