'''Need Internet Connection (don't run on remote server)'''

from torchvision.models import resnet50
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import eagerpy as ep

def main() -> None:
  model = resnet50(weights='DEFAULT')
  model.eval()

  preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
  fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

  

  images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
  clean_acc = accuracy(fmodel, images, labels)
  print(f"clean accuracy:  {clean_acc * 100:.1f} %")

  attack = LinfPGD()
  epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
  ]
  raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
  robust_accuracy = 1 - success.float32().mean(axis=-1)
  print("robust accuracy for perturbations with")

  for eps, acc in zip(epsilons, robust_accuracy):
      print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

if __name__ == "__main__":
   main()





