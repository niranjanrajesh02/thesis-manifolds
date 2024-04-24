from data import get_classes, get_class_data, id_to_index, id_to_label 
from get_model import load_model
import torch
import numpy as np
import torchvision
import pickle

import os
from dotenv import load_dotenv

load_dotenv()
pickle_path = os.getenv("PICKLE_DATA_PATH")

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
    print(acc)
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
        class_dl, class_label, class_index = get_class_data(class_ids_paths[i], class_ids[i])
        print("Class: ",class_label, "Index: ", class_index)
        class_act = get_activations(r50, class_dl)
        print(class_label, class_act.shape)
        class_activations[class_index] = class_act
    
    with open(os.path.join(pickle_path, 'r50_class_activations.pkl'), 'wb') as f:
        pickle.dump(class_activations, f)
    return

def get_class_accuracies(model, class_ids, class_ids_paths):
    class_accuracies = {}
    for i in range(len(class_ids)):
        class_dl, class_label, class_index = get_class_data(class_ids_paths[i], class_ids[i])
        print("Class: ",class_label, "Index: ", class_index)
        class_acc = model_evaluate(r50, class_dl)
        print(class_label, class_acc)
        class_accuracies[class_index] = class_acc
    
    with open(os.path.join(pickle_path, './r50_class_accuracies.pkl'), 'wb') as f:
        pickle.dump(class_accuracies, f)
    return

if __name__ == '__main__':
    class_ids, class_ids_paths = get_classes()
    r50 = load_model()
    r50.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ",device)
    r50.to(device)

    get_class_activations(r50, class_ids, class_ids_paths)
    print("Activations saved in a pickle file")
    
    get_class_accuracies(r50, class_ids, class_ids_paths)
    print("Accuracies saved in a pickle file")




    

    
