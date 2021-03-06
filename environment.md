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

## Create an Anaconda environment
More details [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).
```
conda create --name tensorflow
```

## Install TensorFlow

Follow the instructions in the [Installing with Anaconda section](https://www.tensorflow.org/install/install_linux#InstallingAnaconda) of TensorFlow docs.

Make sure you have the environment activated:
```
source activate tensorflow
```

Install TensorFlow via pip:
```
(tensorflow)$ pip install --ignore-installed --upgrade tfBinaryURL
```

You have to choose the tfBinaryURL to match the TensorFlow version required for your experiments. For example, to install version 1.8.0 with GPU support use this command:
```
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp34-cp34m-linux_x86_64.whl
```

## Configuring CUDA and CuDNN
Each TensorFlow version requires specific of CUDA and CuDNN versions (see a complete list [here](https://www.tensorflow.org/install/install_sources)).

To choose specific versions of CUDA and CuDNN you need to configure the environment variables `LD_LIBRARY_PATH` (CUDA) and `DYLD_LIBRARY_PATH` (CuDNN). An easy way to set these variables is using Anaconda's activation/deactivation scripts (more details [here](https://blog.kovalevskyi.com/multiple-version-of-cuda-libraries-on-the-same-machine-b9502d50ae77)). To create an activation script execute the following commands:
```
mkdir -p ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/activate.d
touch ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/activate.d/activate.sh
chmod +x ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/activate.d/activate.sh
vim ~/<ANACONDA_HOME>/envs/<tensorflow_env>/etc/conda/activate.d/activate.sh
```

Add the following contents (on the Cohort machines, `CUDA_HOME` and `CUDNN_HOME` will probably be in the `/opt` folder):
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

## Screen commands
### Log terminal output
From this [stackoverflow question](https://stackoverflow.com/questions/14208001/save-screen-program-output-to-a-file).

> `Ctrl+a` then `Shift+h`. You can view the file `screenlog.0` while the program is still running.

> If the log can’t be created, then try changing the screen window’s working directory: `Ctrl+a + :` and type for example `chdir /home/foobar/baz`
