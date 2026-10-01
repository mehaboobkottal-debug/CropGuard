import sys
import os

sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import create_dataloaders
from src.models.model import create_model


def main():
    print("CropGuard Training Pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, num_classes = create_dataloaders(
        "data/raw",
        batch_size=32
    )

    model = create_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss/len(train_loader):.4f}"
        )

    torch.save(model.state_dict(), "cropguard_model.pth")

    print("Model saved: cropguard_model.pth")


if __name__ == "__main__":
    main()