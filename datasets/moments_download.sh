#python download.py http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Mini.zip --unzip

# Moments in Time Mini training and validation set
wget -c -nc http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Mini.zip
unzip -n Moments_in_Time_Mini -d moments_data

# Moments in Time Mini test set
wget -c -nc http://data.csail.mit.edu/soundnet/actions3/split1/momentsMiniTest.zip
unzip -n momentsMiniTest -d moments_test