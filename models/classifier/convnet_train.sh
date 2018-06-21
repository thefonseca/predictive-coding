mkdir -p logs
python convnet.py --config moments_2c_transfer_kitti_R3 #2>&1 | tee "./logs/moments_2c_transfer_kitti_R3_$(date +"%FT%H%M").log"