'''Need Internet Connection (don't run on remote server)'''

from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD
import eagerpy as ep
from get_model import load_model
from data import get_classes, get_class_data
import numpy as np
import os
import pickle
from dotenv import load_dotenv
import argparse

load_dotenv()
save_path = os.getenv("PICKLE_DATA_PATH")

def setup_attack():
  preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
  fmodel = PyTorchModel(load_model().eval(), bounds=(0, 1), preprocessing=preprocessing)
  attack = LinfPGD(abs_stepsize=0.01, steps=20)

  return attack, fmodel

def get_adv_acc(class_dl, attack, fmodel, eps) -> float:
  fooled = []
  for images, labels in class_dl:
    images, labels = ep.astensors(images, labels)
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=[eps])
    fooled = np.append(fooled, success.float32().numpy()[0])

  adv_acc = (1 - np.mean(fooled)) * 100
  return adv_acc

def main(class_ids, class_ids_paths) -> None:
  
  attack, fmodel = setup_attack()

  epsilons = [0, 0.0002,0.0005,0.002, 0.005]
  adv_accuracies = {}
  for i in range(len(class_ids)):
    class_dl, class_label, class_index = get_class_data(class_ids[i], class_ids_paths[i], bs=50, transform=False)
    print(f"Generating attacks for class: {class_label} ({i}/{len(class_ids)})")
    adv_accuracies[class_index] = []
    for eps in epsilons:
      print(f'Running attack for epsilon: {eps} ...')
      adv_acc = get_adv_acc(class_dl, attack, fmodel, eps)
      adv_accuracies[class_index].append(adv_acc) 

  print('Adv Accuracies:', adv_accuracies)
  # save adv_accuracies as a pickle 
  file_name = f'r50_adv_accuracies_{len(class_ids)}c.pkl'
  with open(os.path.join(save_path, file_name), 'wb') as f:
    pickle.dump(adv_accuracies, f)
  print(f'Adv Accuracies Saved in {file_name}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", type=str, default='imagenette', help="Dataset to use for training (ImageNet or Imagenette)")
  parser.add_argument("--n_classes", type=int, default=10, help="Number of classes to evaluate class-wise robustness")
  parser.add_argument("--train", type=bool, default=False, help="Whether to use train or validation set")
  args = parser.parse_args()
  class_ids, class_ids_paths = get_classes(is_train=args.train, imgnet=args.dataset, rand_subset=args.n_classes)
  main(class_ids, class_ids_paths)





