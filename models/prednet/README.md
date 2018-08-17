# Predictive Coding model

This contains scripts to train and extract from Lotter et al. predictive coding model.

## Relevant files
* [evaluate.py](./evaluate.py): script to extract predictive coding features and generate frame predictions.

Example: extract features using a PredNet with random weights for data from 3 classes of the Moments in Time dataset (defined in settings.py)
```
> python evaluate.py prednet_random_finetuned_moments__representation --task 3c
```

* [train.py](./train.py): script to train the predictive coding neural network.
Example: train PredNet from scratch using data from 10 classes of the Moments in Time dataset (defined in settings.py)
```
> python evaluate.py prednet_random_finetuned_moments__representation --task 10c
```

* [settings.py](./settings.py): parameters for each experiment.
