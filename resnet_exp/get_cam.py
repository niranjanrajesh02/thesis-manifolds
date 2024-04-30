from data import get_classes, get_class_data
from get_model import load_model
import torch
import numpy as np
import pickle
from sklearn.decomposition import PCA
import os
from dotenv import load_dotenv
import argparse
from data import id_to_index

load_dotenv()
save_path = os.getenv("PICKLE_DATA_PATH")



def get_activations(model, data):
    activations = []

    # wherever there is a relu, we want to save the activations
    def activation_hook(module, input, output):
        # print(module, output.shape)
        activations.append(output.detach().cpu().numpy().reshape(output.shape[0], -1))
        return


    # register the hook on the last layer
    model.avgpool.register_forward_hook(activation_hook)

    # run the model
    with torch.no_grad():
        for images, labels in data:
            outputs = model(images)
            # we only need to run the model once to get the activations
        
    # convert the activations to a numpy array
    return  np.concatenate(activations, axis=0)

def get_class_activations(model, class_ids, class_ids_paths):
    class_activations = {}
    for i in range(len(class_ids)):
        class_dl, class_label, class_index = get_class_data(class_ids[i],class_ids_paths[i], bs=50)
        print("Class: ",class_label, "Index: ", class_index)
        class_act = get_activations(model, class_dl)
        print(class_label, class_act.shape)
        class_activations[class_index] = class_act

    file_path = os.path.join(save_path, f'r50_class_activations_{len(class_ids)}c.pkl') 
    with open(os.path.join(save_path, file_path), 'wb') as f:
        pickle.dump(class_activations, f)
    return



def estimate_linear_dim_PCA(activations, threshold=0.95):
    pca = PCA()
    pca.fit(activations)
    # Find the number of components required to explain 95% of the variance
    var = np.cumsum(pca.explained_variance_ratio_)
    # print("Cumulative variance: ", var)
    linear_dim = np.argmax(var > threshold) + 1
    # print("Linear dimension: ", linear_dim)
    return linear_dim

def get_class_manifold_dims(act_name):
    activations = pickle.load(open(os.path.join(save_path, act_name), 'rb'))
    print(activations)
    class_dims = {}
    for key in activations.keys():
        class_dims[key] = estimate_linear_dim_PCA(activations[key])
    
    file_name = f'r50_class_manifold_dims_{len(class_dims)}.pkl'

    with open(os.path.join(save_path, file_name), 'wb') as f:
        pickle.dump(class_dims, f)
    
    print(class_dims)
    return class_dims

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Class Activation Maps')
    parser.add_argument('--train', action='store_true', help='Train the model', default=False)
    parser.add_argument('--task', type=str, help='Task to perform')
    parser.add_argument('--dataset', type=str, help='Dataset to use', default='imagenette')
    parser.add_argument('--n_classes', type=int, help='Number of classes to consider', default=10)
    parser.add_argument('--act_name', type=str, help='File name of activations')
    args = parser.parse_args()

    if args.task == 'activations':
        # assert args.train == True
        class_ids, class_ids_paths = get_classes(is_train=args.train, imgnet=args.dataset, rand_subset=args.n_classes)
        r50 = load_model()
        r50.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ",device)
        r50.to(device)
        
        with open(os.path.join(save_path, 'r50_adv_accuracies_50c.pkl'), 'rb') as f:
            adv_accuracies = pickle.load(f)
        
        # class_inds = [id_to_index(cid) for cid in class_ids]
        get_class_activations(r50, class_ids, class_ids_paths)
        print("Activations saved in a pickle file")
    
    elif args.task == 'manifold_dims':
        get_class_manifold_dims(args.act_name)
        print("Manifold dimensions saved in a pickle file")
    






    

    
