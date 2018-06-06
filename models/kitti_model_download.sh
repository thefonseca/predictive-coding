savedir="./model_data/kitti_keras"
mkdir -p -- "$savedir"
wget https://www.dropbox.com/s/z7ittwfxa5css7a/model_data_keras2.zip?dl=0 -O $savedir/kitti_keras.zip
unzip -j $savedir/kitti_keras.zip -d $savedir
rm $savedir/kitti_keras.zip
