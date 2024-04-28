import os
from dotenv import load_dotenv
import random
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import cv2
from torchvision.io import read_image, ImageReadMode
import torch
from torchvision.models import ResNet50_Weights

load_dotenv()
DATA_PATH = os.getenv("IMAGENETTE_PATH")
# DATA_PATH = os.getenv("IMAGEWOOF_PATH")
labels_path = os.getenv("IMAGENET_LABELS_PATH")
imagenette_labels_path = os.getenv("IMAGENETTE_LABELS_PATH")
random.seed(42)
r50_transforms = ResNet50_Weights.DEFAULT.transforms()
r50_transforms = transforms.Compose([
     transforms.ToPILImage(),
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
    #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
 ])



def get_labels():
  # read json file as dict
  with open(labels_path, "r") as f:
    labels = json.load(f)
  # remove "-n" from each value["id"]
  for key, value in labels.items():
    value["id"] = value["id"].split("-")[0]
    value["id"] = str("n" + value["id"])
  
  #* value["id"] => ID / folder name
  #* value["label"] => label
  return labels


class CAMdataset(Dataset):
  def __init__(self, root_dir, class_ind, transform=True):
    self.root_dir = root_dir
    self.transform = r50_transforms
    self.images = os.listdir(root_dir)
    self.class_index = class_ind
  
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    img_path = os.path.join(self.root_dir, self.images[idx])
    image = read_image(img_path, ImageReadMode.RGB).float()
    if self.transform:
      image = self.transform(image)

    return image, self.class_index


def get_class_data(class_id, class_path, bs=1, transform=True):
  class_index = id_to_index(class_id)
  class_label = id_to_label(class_id)
  class_ds = CAMdataset(class_path, class_index, transform=transform)
  class_dl = DataLoader(class_ds, batch_size=bs, shuffle=False)
  return class_dl, class_label, class_index

def get_classes(is_train=True):
  if is_train:
    new_data_path = os.path.join(DATA_PATH, "train")
  else:
    new_data_path = os.path.join(DATA_PATH, "val")
  # total images found in all classes
  total_images = 0
  data = "Train" if is_train else "Val"
  print(f"Getting {data} Images...")
  for class_id in os.listdir(new_data_path):
    class_path = os.path.join(new_data_path, class_id)
    class_images = os.listdir(class_path)
    print(f"{id_to_label(class_id)}: {len(class_images)}")
    total_images += len(class_images)
  class_ids = os.listdir(new_data_path)
  class_ids_paths = [os.path.join(new_data_path, class_id) for class_id in class_ids]
  return class_ids, class_ids_paths

def id_to_index(class_id):
  labels = get_labels()
  return int([key for key,value in labels.items() if value["id"] == class_id][0])

def id_to_label(class_id):
  labels = get_labels()
  return [value["label"].split(",")[0].split(" ")[-1] for key,value in labels.items() if value["id"] == class_id][0]

def index_to_label(class_index):
  # print(class_index)
  labels = get_labels()
  return [value["label"].split(",")[0].split(" ")[-1] for key,value in labels.items() if int(key) == int(class_index)][0]

if __name__ == "__main__":
  labels = get_labels()
  # print(labels)
  class_ids, class_ids_paths= get_classes(is_train=False)
  # compute label key when label[key]["id"] == class_id
  # print(class_ids[1])
  # class_ind = id_to_index(class_ids[1])
  # class_label = id_to_label(class_ids[1])
  # print(class_ids[1], class_ind, class_label, class_ids_paths[1])
  # class_data = get_class_data(class_ids_paths[1], class_ids[1])
  # print(class_data)
  






