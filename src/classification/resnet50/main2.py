import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Same device and class count as during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 27

# Define model architecture (must match training)
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

