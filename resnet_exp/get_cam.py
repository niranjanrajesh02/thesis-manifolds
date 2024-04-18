from data import get_classes, get_class_data, id_to_index, id_to_label 
from get_model import load_model
import torch


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



if __name__ == '__main__':
    class_ids, class_ids_paths = get_classes()
    class_ds = get_class_data(class_ids_paths[1], class_ids[1])

    r50 = load_model()
    r50.eval()
    model_evaluate(r50, class_ds)