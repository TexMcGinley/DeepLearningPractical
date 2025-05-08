import torch
from torchvision.models import alexnet
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
import os
import numpy as np
from utils.alphabet import alphabet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 27

model = alexnet()
model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(alphabet))
model.load_state_dict(torch.load("alexnet_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run(args):
    os.makedirs(args.model_output, exist_ok=True)
    for text in os.listdir(args.model_input):

        # Empty the existing file if it alr exists
        with open(f"{args.model_output}/{text}.txt", "w"):
            pass

        for line in os.listdir(f"{args.model_input}/{text}"):
            images = []
            for char in os.listdir(f"{args.model_input}/{text}/{line}"):
                image_path = f"{args.model_input}/{text}/{line}/{char}"
                image = Image.open(image_path).convert("RGB")
                image = ImageOps.invert(image)
                image = transform(image)

                image = image.to(device)
                images.append(image)

            batch = torch.stack(images).to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                
                with open(f"{args.model_output}/{text}.txt", "a", encoding='utf-8') as f:
                    chars =np.array(list(alphabet.values()))[predicted.cpu().numpy()][::-1]
                    chars = [char.encode().decode('unicode_escape') for char in chars ]
                    f.write("".join(chars)+"\n")






