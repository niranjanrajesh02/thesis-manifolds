import torch
from sklearn.decomposition import PCA
from torchvision.models import resnet50
from dotenv import load_dotenv
import os
load_dotenv()

def load_model():
    model_path = os.getenv('RESNET50_PATH') 
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(model_path)) # Load the model
    print("Model loaded")
    # print(model)
    # print(model.children())
    return model

def load_mult_model(model_name):
  weight_paths = {
  "ResNet50": os.getenv("RESNET50_PATH"),
  "AlexNet": os.getenv("ALEXNET_PATH"),
  "ConvNeXt": os.getenv("CONVNEXT_PATH"),
  "DenseNet": os.getenv("DENSENET_PATH"),
  "VGG16": os.getenv("VGG16_PATH"),
  "ViT": os.getenv("VIT_PATH"),
  "RobustResNet50": os.getenv("ROBUST_RESNET_PATH"),
  "Robust2ResNet50": os.getenv("ROBUST2_RESNET_PATH")
    }
  
  if model_name not in weight_paths:  
    raise ValueError(f"Model {model_name} not found")
  
  model_weights = torch.load(weight_paths[model_name], map_location=torch.device('cpu'))
  
  if model_name == "ResNet50":
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    model.load_state_dict(model_weights)
    print(f"Loaded {model_name} model")
  
  elif model_name == "AlexNet":
    from torchvision.models import alexnet
    model = alexnet(weights=None)
    model.load_state_dict(model_weights)
    print(f"Loaded {model_name} model")

  elif model_name == "ConvNeXt":
    from torchvision.models.convnext import convnext_base
    model = convnext_base(weights=None)
    model.load_state_dict(model_weights)
    print(f"Loaded {model_name} model")
  
  elif model_name == "DenseNet":
    from torchvision.models import densenet161
    model = densenet161(weights=None) 
    model.load_state_dict(model_weights)
    print(f"Loaded {model_name} model")

  elif model_name == "VGG16":
    from torchvision.models import vgg16
    model = vgg16(weights=None)
    model.load_state_dict(model_weights)
    print(f"Loaded {model_name} model")


  elif model_name == "ViT":
    from torchvision.models.vision_transformer import vit_b_32
    model = vit_b_32(weights=None)
    model.load_state_dict(model_weights)
    print(f"Loaded {model_name} model")
  
  elif model_name == "RobustResNet50":
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    model.load_state_dict(model_weights)
    print(f"Loaded {model_name} model")
    # print(model)
  
  elif model_name == "Robust2ResNet50":
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    model.load_state_dict(model_weights)
    print(f"Loaded {model_name} model")
    print(model)
    
  # print(model)
  return model


if __name__ == "__main__":
    # load_model()
    load_mult_model("Robust2ResNet50")
