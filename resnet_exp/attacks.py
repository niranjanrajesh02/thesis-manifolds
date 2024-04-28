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

load_dotenv()
save_path = os.getenv("PICKLE_DATA_PATH")

def setup_attack():
  preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
  fmodel = PyTorchModel(load_model().eval(), bounds=(0, 1), preprocessing=preprocessing)
  attack = LinfPGD(abs_stepsize=0.01, steps=20)

  return attack, fmodel

def get_adv_acc(class_dl, attack, fmodel) -> float:
  fooled = []
  for images, labels in class_dl:
    images, labels = ep.astensors(images, labels)
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=[0.005])
    fooled = np.append(fooled, success.float32().numpy()[0])

  adv_acc = (1 - np.mean(fooled)) * 100
  return adv_acc

def main() -> None:
  class_ids, class_ids_paths = get_classes(is_train=False)
  attack, fmodel = setup_attack()

  adv_accuracies = {}
  for i in range(len(class_ids)):
    class_dl, class_label, class_index = get_class_data(class_ids[i], class_ids_paths[1], bs=50, transform=False)
    adv_acc = get_adv_acc(class_dl, attack, fmodel)
    adv_accuracies[class_index] = adv_acc

  print('Adv Accuracies:', adv_accuracies)
  # save adv_accuracies as a pickle file
  with open(save_path + 'r50_adv_accuracies.pkl', 'wb') as f:
    pickle.dump(adv_accuracies, f)
  print('Adv Accuracies Saved in r50_adv_accuracies.pkl')


if __name__ == "__main__":
   main()





