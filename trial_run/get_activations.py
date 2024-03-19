import tensorflow as tf
import numpy as np
import os
import keras
from data import get_cifar_subset


def get_model(path='/home/niranjan.rajesh_asp24/pretrained_models/', model_name='effnet'):
    assert model_name in ['effnet', 'mobnet', 'resnet50'], "Invalid model name"
    base_model = keras.models.load_model(os.path.join(path, f'{model_name}.h5'))
    for layer in base_model.layers:
        layer.trainable = False
    head = base_model.output
    head = keras.layers.Flatten()(head)
    # layer name
    head = keras.layers.Dense(256, activation='relu', name='final_dense')(head)
    head = keras.layers.Dropout(0.5)(head)
    head = keras.layers.Dense(10, activation='softmax')(head)
    model = keras.models.Model(inputs=base_model.input, outputs=head)
    return model



def get_data():
    train_ds, val_ds = get_cifar_subset(class_name='dog')
    return train_ds, val_ds

def get_activations(model, ds):
    
    activations = []
    layer_counter = 0
    for layer in model.layers:

        # looping through all conv layers in each block
        # TODO: Add support for other CNN naming convention
        if ('block' in layer.name) and ('activation' in layer.name) and (not 'expand' in layer.name):
            print(f"Getting Activations for {layer.name}")
            intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                         outputs=layer.output)
            intermediate_activations = intermediate_layer_model.predict(ds, verbose=0)
            intermediate_activations = intermediate_activations.reshape(intermediate_activations.shape[0], -1)
            print(f"Intermediate Activations shape {intermediate_activations.shape}")
            activations.append(intermediate_activations)
            del intermediate_layer_model
            del intermediate_activations
            layer_counter += 1
           
    
    num_cols = sum(activation.shape[1] for activation in activations)
    final_activations = np.zeros((activations[0].shape[0], num_cols))

    col = 0
    for activation in activations:
        final_activations[:, col:col+activation.shape[1]] = activation
        col += activation.shape[1]

    
    print(f"Got activations for {layer_counter} layers")
    print(f"Activations shape {final_activations.shape}")
    
    # outputs N x D matrix where N is the number of images and D is the number of features
    return final_activations

if __name__ == '__main__':
    model = get_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())    
    
    train_ds, val_ds = get_data()
    # model.fit(train_ds, epochs=1, validation_data=val_ds, verbose=1)
    
    # Get the activations
    activations = get_activations(model, train_ds)
    # save activations
    np.save('/home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/results/activations.npy', activations)

    # load activations
    activations = np.load('/home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/results/activations.npy')
    print(activations.shape)




