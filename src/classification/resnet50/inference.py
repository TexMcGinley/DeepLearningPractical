import torch
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
import os
import numpy as np
from src.utils.alphabet import alphabet

# Code modified from other university projects as specified in the README file
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 27

    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("models/resnet50_model.pth", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    os.makedirs("results", exist_ok=True)
    for i, text in enumerate(os.listdir("outputs/char-crops")):

        # Empty the existing file if it alr exists
        with open(f"results/img_{i:03}_characters.txt", "w"):
            pass

        for line in os.listdir(f"outputs/char-crops/{text}"):
            images = []
            for char in os.listdir(f"outputs/char-crops/{text}/{line}"):
                image_path = f"outputs/char-crops/{text}/{line}/{char}"
                image = Image.open(image_path).convert("RGB")
                image = ImageOps.invert(image)
                image = transform(image)

                image = image.to(device)
                images.append(image)

            batch = torch.stack(images).to(device)

            
            with torch.no_grad():
                model.eval()
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                
                with open(f"results/img_{i:03}_characters.txt", "a", encoding='utf-8') as f:
                    chars =np.array(list(alphabet.values()))[predicted.cpu().numpy()][::-1]
                    chars = [char.encode().decode('unicode_escape') for char in chars ]
                    f.write("".join(chars)+"\n")






