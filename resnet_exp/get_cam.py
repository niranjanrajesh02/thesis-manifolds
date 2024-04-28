from data import get_classes, get_class_data
from get_model import load_model
import torch
import numpy as np
import pickle
from sklearn.decomposition import PCA
import os
from dotenv import load_dotenv
import argparse


load_dotenv()
save_path = os.getenv("PICKLE_DATA_PATH")

def model_evaluate(model, data):
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
      for images, labels in data:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc

def get_activations(model, data):
    activations = []

    # wherever there is a relu, we want to save the activations
    def activation_hook(module, input, output):
        # print(module, output.shape)
        activations.append(output.detach().cpu().numpy().reshape(output.shape[0], -1))
        return

    # register the hook on conv3 of each the third bottleneck of each layer
    # def register_hooks(module):
    #     # print the children of module
    #     bottle_necks = 0
    #     for child in module.children():
    #         if isinstance(child, torchvision.models.resnet.Bottleneck):
    #             bottle_necks += 1
    #             # register the hook on conv3
    #             if bottle_necks == 3:
    #                 child.conv3.register_forward_hook(activation_hook)
    #         print(child)

    # register the hook
    # register_hooks(model.layer1)
    # register_hooks(model.layer2)
    # register_hooks(model.layer3)
    # register_hooks(model.layer4)

    # register the hook on the last layer
    model.avgpool.register_forward_hook(activation_hook)

    # run the model
    counter = 0
    with torch.no_grad():
        for images, labels in data:
            outputs = model(images)
            # we only need to run the model once to get the activations
        
    # convert the activations to a numpy array
    return  np.concatenate(activations, axis=0)

def get_class_activations(model, class_ids, class_ids_paths):
    class_activations = {}
    for i in range(len(class_ids)):
        class_dl, class_label, class_index = get_class_data(class_ids[i],class_ids_paths[i])
        print("Class: ",class_label, "Index: ", class_index)
        class_act = get_activations(r50, class_dl)
        print(class_label, class_act.shape)
        class_activations[class_index] = class_act
    
    with open(os.path.join(save_path, 'r50_class_activations.pkl'), 'wb') as f:
        pickle.dump(class_activations, f)
    return

def get_class_accuracies(model, class_ids, class_ids_paths, train=True):
    class_accuracies = {}
    for i in range(len(class_ids)):
        class_dl, class_label, class_index = get_class_data(class_ids[i], class_ids_paths[i])
        print("Class: ",class_label, "Index: ", class_index)
        class_acc = model_evaluate(r50, class_dl)
        print(class_label, class_acc)
        class_accuracies[class_index] = class_acc
    
   
    with open(os.path.join(save_path, f'r50_class_{"train_" if train else "valid_"}accuracies.pkl'), 'wb') as f:
        pickle.dump(class_accuracies, f)

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

def get_class_manifold_dims():
    activations = pickle.load(open(os.path.join(save_path, 'r50_class_activations.pkl'), 'rb'))
    print(activations)
    class_dims = {}
    for key in activations.keys():
        class_dims[key] = estimate_linear_dim_PCA(activations[key])
    with open(os.path.join(save_path, 'r50_class_manifold_dims.pkl'), 'wb') as f:
        pickle.dump(class_dims, f)
    
    print(class_dims)
    return class_dims

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Class Activation Maps')
    parser.add_argument('--train', action='store_true', help='Train the model', default=False)
    parser.add_argument('--task', type=str, help='Task to perform')
    args = parser.parse_args()

    class_ids, class_ids_paths = get_classes(is_train=args.train)

    if args.task == 'accuracies':
        r50 = load_model()
        r50.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ",device)
        r50.to(device)
        get_class_accuracies(r50, class_ids, class_ids_paths, train=args.train)
        print("Accuracies saved in a pickle file")
    
    elif args.task == 'activations':
        assert args.train == True
        r50 = load_model()
        r50.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ",device)
        r50.to(device)
        get_class_activations(r50, class_ids, class_ids_paths)
        print("Activations saved in a pickle file")
    
    elif args.task == 'manifold_dims':
        get_class_manifold_dims()
        print("Manifold dimensions saved in a pickle file")
    






    

    
