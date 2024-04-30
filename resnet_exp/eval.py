import torch
import numpy as np
from data import get_class_data, get_classes
import pickle
import os
import dotenv
from get_model import load_model
import argparse

dotenv.load_dotenv()
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


def get_class_accuracies(model, class_ids, class_ids_paths, train=True):
    class_accuracies = {}
    for i in range(len(class_ids)):
        class_dl, class_label, class_index = get_class_data(class_ids[i], class_ids_paths[i])
        print("Class: ",class_label, "Index: ", class_index)
        class_acc = model_evaluate(model, class_dl)
        print(class_label, class_acc)
        class_accuracies[class_index] = class_acc
    print(class_accuracies)
   
    with open(os.path.join(save_path, f'r50_class_{"train_" if train else "valid_"}accuracies.pkl'), 'wb') as f:
        pickle.dump(class_accuracies, f)

    return


if __name__ == "__main__":
      
        r50 = load_model()
        r50.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ",device)
        r50.to(device)  

        class_ids, class_ids_paths = get_classes(is_train=False)
        print("Class IDs: ", class_ids)
        print("Class Paths: ", class_ids_paths)
        get_class_accuracies(r50, class_ids, class_ids_paths, train=False)
        print("Accuracies saved in a pickle file")