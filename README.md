# My Capstone Thesis Code Repository
## Thesis Topic: Understanding CNN Robustness with Class Activation Manifolds


### ```resnet_exp/```
This directory contains the code, plots and data for experiments on an ImageNet-Pretrained Resnet50. 
- ```data.py``` loads in data from two subsets of ImageNet - Imagenette and Imagewoof ([source](https://github.com/fastai/imagenette))
- ```eval.py``` evaluates the model on the data and generates classwise accuracies
- ```get_cam.py``` extracts the activations of the model and estimates the dimensionality of the Class Activation Manifold
- ```attacks.py``` uses [FoolBox](https://github.com/bethgelab/foolbox) to generate adversarial attacks and evaluates model's robustness
- ```adv_train.ipynb``` adversarially trains a CNN on MNIST dataset using [Adversarial Robustness Toolkit](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- ```get_model.py``` loads the CNNs necessary for experimentation
- ```make_plots.py``` generates all plots used in my presentation and report

# All experiments were run on Ashoka HPC. Contact me for any more information needed for reproducing my results!
  

  


