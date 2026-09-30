import sys
import os

sys.path.append(os.path.abspath("."))

from src.data.dataset import load_dataset

dataset = load_dataset("data/raw")

print("Total Images:", len(dataset))
print("Classes:", dataset.classes)
print("Number of Classes:", len(dataset.classes))