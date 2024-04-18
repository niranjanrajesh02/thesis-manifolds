import torch
from torchvision.models import resnet50
from dotenv import load_dotenv
import os
load_dotenv()


model_path = os.getenv('RESNET50_PATH')

def load_model():
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(model_path)) # Load the model
    print("Model loaded")
    # print(model)
    return model

if __name__ == "__main__":
    load_model()
