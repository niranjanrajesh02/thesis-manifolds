# My Capstone Thesis Code Repository
## Thesis Topic: Investigating Neural Manifold Dimensionality in CNNs

### ```trial_run/```
This directory contains my initial attempts to extract activations of CNNs on a CIFAR10 dataset and estimate the dimensionality of the manifold they create. This was written in TensorFlow

### ```resnet_exp/```
This directory contains the code, plots and data for experiments on an ImageNet-Pretrained Resnet50. 
- ```data.py``` loads in data from two subsets of ImageNet - Imagenette and Imagewoof ([source](https://github.com/fastai/imagenette))
- ```eval.py``` evaluates the model on the data and generates classwise accuracies
- ```get_cam.py``` extracts the activations of the model and estimates the dimensionality of the Class Activation Manifold
- ```attacks.py``` uses FoolBox to generate adversarial attacks and evaluates model's robustness

  


