import tensorflow as tf
import numpy as np
import os
import keras
from data import get_cifar_subset


def get_model(path='/home/niranjan.rajesh_asp24/pretrained_models/', model_name='effnet'):
    assert model_name in ['effnet', 'mobnet', 'resnet50'], "Invalid model name"
    model = keras.models.load_model(os.path.join(path, f'{model_name}.h5'))
    return model

def get_data():
    train_ds, val_ds = get_cifar_subset()
    return train_ds, val_ds

if __name__ == '__main__':
    model = get_model()
    train_ds, val_ds = get_data()

    for image, label in train_ds.take(1):
      print("Image shape: ", image.numpy().shape)
      print("Label: ", label.numpy())


