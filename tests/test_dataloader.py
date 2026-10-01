import sys
import os

sys.path.append(os.path.abspath("."))

from src.data.dataloader import create_dataloaders

train_loader, val_loader, num_classes = create_dataloaders(
    "data/raw",
    batch_size=32
)

print("Train Batches:", len(train_loader))
print("Validation Batches:", len(val_loader))
print("Classes:", num_classes)