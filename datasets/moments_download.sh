#python download.py http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Mini.zip --unzip

# Moments in Time Mini training and validation set
wget -c -nc http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Mini.zip

if [ -e moments_data ]
then
    echo "Could not unzip: directory 'moments_data' already exists!"
else
    unzip -n Moments_in_Time_Mini
    echo "Renaming dataset directory to 'moments_data'..."
    mv ./Moments_in_Time_Mini ./moments_data
fi

# Moments in Time Mini test set
wget -c -nc http://data.csail.mit.edu/soundnet/actions3/split1/momentsMiniTest.zip

if [ -e moments_data/test ]
then
    echo "Could not unzip: directory 'moments_data/test' already exists!"
else
    unzip -n momentsMiniTest
    echo "Moving test set to directory 'moments_data'..."
    mv ./momentsMiniTest ./moments_data/test
fi
