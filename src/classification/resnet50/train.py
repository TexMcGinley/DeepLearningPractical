import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm 
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.aggregation import MeanMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.plot import plot
import argparse

# Code modified from other university projects as specified in the README file
def train():
    parser = argparse.ArgumentParser(description="Arguments for training ResNet50 model")
    parser.add_argument("input", type=str, help="Path to the input data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Where to write distorted images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    args = parser.parse_args()

    assert args.input, "Input directory is required"
    assert os.path.exists(args.input), f"Input directory {args.input} does not exist"

    alphabet = os.listdir(args.input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = datasets.ImageFolder(root=args.input, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(alphabet))
    model.to(device)
    summary(model, (3, 48, 48))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer)

    train_loss, test_loss = [], []
    train_acc, test_acc = [], []

    loss_tracker = MeanMetric().to(device)
    acc_tracker = MulticlassAccuracy(num_classes=27).to(device)

    for epoch in range(args.epochs):
        model.train()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item())
            acc_tracker.update(outputs, labels)

        train_loss.append(loss_tracker.compute().item())
        train_acc.append(acc_tracker.compute().item())

        loss_tracker.reset()
        acc_tracker.reset()


        with torch.no_grad():
            model.eval()

            for images, labels in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{args.epochs}"):

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss_tracker.update(loss.item())
                acc_tracker.update(outputs, labels)

            test_loss.append(loss_tracker.compute().item())
            test_acc.append(acc_tracker.compute().item())

            scheduler.step(loss_tracker.compute().item())

            loss_tracker.reset()
            acc_tracker.reset()

            print(f"Epoch {epoch+1}, Loss: {train_loss[-1]:.4f}, Accuracy: {train_acc[-1]*100:.2f}%, Test Loss: {train_loss[-1]:.2f} Test Accuracy: {test_acc[-1]*100:.2f}%")

    plot(train_loss, test_loss, "Loss ResNet50 Model", "Train Loss", "Test Loss")
    plot(train_acc, test_acc, "Accuracy ResNet50 Model", "Train Accuracy", "Test Accuracy")

    torch.save(model.state_dict(), "models/resnet50_model.pth")