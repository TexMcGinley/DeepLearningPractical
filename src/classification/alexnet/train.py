import torch
import torch.nn as nn
from torchvision.models import alexnet
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

def train(args = None):
    alphabet = os.listdir("data/train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = datasets.ImageFolder(root="data/train", transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = alexnet()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(alphabet))
    model.to(device)
    summary(model, (3, 224, 224))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer)

    num_epochs = 10

    train_loss, test_loss = [], []
    train_acc, test_acc = [], []

    loss_tracker = MeanMetric().to(device)
    acc_tracker = MulticlassAccuracy(num_classes=27).to(device)

    for epoch in range(num_epochs):
        model.train()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

            for images, labels in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{num_epochs}"):

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

    plot(train_loss, test_loss, "Loss AlexNet Model", "Train Loss", "Test Loss")
    plot(train_acc, test_acc, "Accuracy AlexNet Model", "Train Accuracy", "Test Accuracy")

    torch.save(model.state_dict(), "models/alexnet_model.pth")