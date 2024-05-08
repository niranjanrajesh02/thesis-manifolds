from art.estimators.classification import PyTorchClassifier
from get_model import load_mult_model

# Load VGG-16
model = load_mult_model('VGG16')

# Load

# Create ART classifier
classifier = PyTorchClassifier(model=model)