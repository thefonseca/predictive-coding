mkdir -p logs

#python moments_evaluate.py --config moments_2c_transfer_kitti --representation --layer 3 --training 2>&1 | tee "./logs/moments_2c_transfer_kitti_train_$(date +"%FT%H%M").log"

python moments_evaluate.py --config moments_2c_transfer_kitti --representation --layer 3 --training 2>&1 | tee "./logs/moments_2c_transfer_kitti_train.log"

python moments_evaluate.py --config moments_2c_transfer_kitti --representation --layer 3 --validation 2>&1 | tee "./logs/moments_2c_transfer_kitti_val.log"