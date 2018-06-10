# Environment configurations

This file documents environment configurations used to run experiments on [the cohort hardware](http://kinloch.inf.ed.ac.uk/cohort/wiki/index.php/Hardware_and_Experiments).

## Create directory
All experiment files are stored in `/disk/ocean/mfonseca`, which we refer to as `EXPERIMENT_HOME`.

## Install Anaconda/Miniconda
From `EXPERIMENT_HOME`, download the version for Linux:
```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
```

The installer will ask to confirm the default install location or to specify an alternate directory. Make sure to specify `EXPERIMENT_HOME`.

## Setting environment variables
The Anaconda installer will prepend Anaconda's location to `PATH` in `.bashrc`. However, when you login via `ssh` this script is not run. To avoid having to `source ~/.bashrc` every time, create a `~/.bash_profile` file with the following contents:

```
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
```

## Configuring CUDA and CuDNN
To choose specific versions of CUDA and CuDNN you need to configure the environment variables `LD_LIBRARY_PATH` (CUDA) and `DYLD_LIBRARY_PATH` (CuDNN). An easy way to set these variables is using Anaconda's activation/deactivation scripts (more details [here](https://blog.kovalevskyi.com/multiple-version-of-cuda-libraries-on-the-same-machine-b9502d50ae77)). To create an activation script execute the following commands:
```
mkdir -p ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/activate.d
touch ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/activate.d/activate.sh
chmod +x ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/activate.d/activate.sh
vim ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/activate.d/activate.sh
```

Add the following contents:
```
#!/bin/sh
ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ORIGINAL_DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=/<CUDA_HOME>/lib64:/<CUDA_HOME>/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/CUDNN_HOME/lib:$DYLD_LIBRARY_PATH
```

And for the deactivation script:
```
mkdir -p ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/deactivate.d
touch ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/deactivate.d/deactivate.sh
chmod +x ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/deactivate.d/deactivate.sh
vim ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/deactivate.d/deactivate.sh
```

```
#!/bin/sh
export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH
unset ORIGINAL_LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$ORIGINAL_DYLD_LIBRARY_PATH
unset ORIGINAL_DYLD_LIBRARY_PATH
```
