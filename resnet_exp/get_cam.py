from data import get_classes, get_class_data
from get_model import load_mult_model
import torch
import numpy as np
import pickle
from sklearn.decomposition import PCA
import os
from dotenv import load_dotenv
import argparse
from dadapy import data
from my_utils import model_namer

load_dotenv()
save_path = os.getenv("PICKLE_DATA_PATH")



def get_activations(model, data_loader, model_name=None):
    activations = []

    # wherever there is a relu, we want to save the activations
    def activation_hook(module, input, output):
        # print(module, output.shape)
        activations.append(output.detach().cpu().numpy().reshape(output.shape[0], -1))
        return
    
    # register the hook on the last layer
    if model_name == 'ResNet50' or model_name == 'VGG16' or model_name == 'ConvNeXt'or model_name == 'RobustResNet50' :
        model.avgpool.register_forward_hook(activation_hook)  
    elif model_name == 'ViT':
        model.encoder.ln.register_forward_hook(activation_hook) 
    elif model_name == 'DenseNet':
        model.features.norm5.register_forward_hook(activation_hook)
    else:
        raise ValueError("Model not supported")

    # run the model
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            # we only need to run the model once to get the activations
        
    # convert the activations to a numpy array
    return  np.concatenate(activations, axis=0)

def get_class_activations(model_name, class_ids, class_ids_paths, is_train=False):

    model = load_mult_model(model_name)
    model.eval()

    class_activations = {}
    for i in range(len(class_ids)):
        class_dl, class_label, class_index = get_class_data(class_ids[i],class_ids_paths[i], bs=50)
        print("Class: ",class_label, "Index: ", class_index)
        class_act = get_activations(model, class_dl, model_name=model_name)
        print(class_label, class_act.shape)
        class_activations[class_index] = class_act

    model_name = model_namer(model_name)
    ds = 'train_' if is_train else ''

    file_path = os.path.join(save_path, f'{model_name}_{ds}class_activations_{len(class_ids)}c.pkl') 
    with open(os.path.join(save_path, file_path), 'wb') as f:
        pickle.dump(class_activations, f)
    return



def estimate_linear_dim_PCA(activations, threshold=0.9):
    pca = PCA()
    pca.fit(activations)
    # Find the number of components required to explain 95% of the variance
    var = np.cumsum(pca.explained_variance_ratio_)
    # print("Cumulative variance: ", var)
    linear_dim = np.argmax(var > threshold) + 1
    # print("Linear dimension: ", linear_dim)

    return linear_dim

def estimate_dim_nn(activations, threshold=0.9):
    # Implement the nearest neighbour based dimensionality estimation
    acts = data.Data(activations)
    dim, err, r = acts.compute_id_2NN()
    # print(dim,err,r)
    return dim


def get_class_manifold_dims(model_name, num_classes, method='PCA', is_train=False):
    model_name = model_namer( model_name)

    file_path = os.path.join(save_path, f'{model_name}_class_activations_{str(num_classes)}c.pkl')
    activations = pickle.load(open(file_path, 'rb'))
    # print(activations)
    class_dims = {}
    for key in activations.keys():
        if method == 'PCA':
            class_dims[key] = estimate_linear_dim_PCA(activations[key])
        elif method == 'NN':
            class_dims[key] = estimate_dim_nn(activations[key])
        
    ds = 'train_' if is_train else ''
    method = '_nn' if method == 'NN' else ''
    file_name = f'{model_name}_{ds}class_manifold_dims_{len(class_dims)}{method}.pkl'

    with open(os.path.join(save_path, file_name), 'wb') as f:
        pickle.dump(class_dims, f)
    
    print(class_dims)
    return class_dims

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Class Activation Maps')
    parser.add_argument('--train', help='Train the model', default=False)
    parser.add_argument('--task', type=str, help='Task to perform')
    parser.add_argument('--dataset', type=str, help='Dataset to use', default='imagenette')
    parser.add_argument('--n_classes', type=int, help='Number of classes to consider', default=10)
    parser.add_argument("--many_models", type=bool, default=False, help="Whether to use many models or not")
    args = parser.parse_args()

    class_ids, class_ids_paths = get_classes(is_train=args.train, imgnet=args.dataset, rand_subset=args.n_classes)
    if args.task == 'activations':
        # assert args.train == True
        if args.many_models:
            models = ["ResNet50"]
            for model in models:
                get_class_activations(model, class_ids, class_ids_paths)
                print(f"Activations for {model} saved in a pickle file")    
        else:
            get_class_activations('Resnet50', class_ids, class_ids_paths)
            print("Activations saved in a pickle file") 
    
    elif args.task == 'manifold_dims':
        if args.many_models:
            models = ["RobustResNet50", "ViT", "VGG16"]
            for model in models:
                get_class_manifold_dims(model, args.n_classes, method='NN')
                print(f"Manifold dimensions for {model} saved in a pickle file")
                
        else:
            get_class_manifold_dims('Resnet50', args.n_classes)
            print("Manifold dimensions saved in a pickle file")
    
    elif args.task == 'both':
        if args.many_models:
            models = [ "VGG16", "DenseNet", "ConvNeXt", "RobustResNet50"]
        
            for model in models:
                get_class_activations(model, class_ids, class_ids_paths, is_train=args.train)
                print(f"Activations for {model} saved in a pickle file") 

                get_class_manifold_dims(model, args.n_classes, is_train=args.train, method='NN')
                print(f"Manifold dimensions for {model} saved in a pickle file")
    
    else:
        raise ValueError("Task not supported")