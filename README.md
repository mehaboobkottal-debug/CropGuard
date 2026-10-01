# CropGuard 🌱

AI-powered Tomato Disease Detection using Deep Learning and PyTorch.

## Project Overview

CropGuard is a computer vision project that detects tomato leaf diseases using a ResNet18 deep learning model trained on the PlantVillage dataset.

## Features

- Tomato disease classification
- ResNet18 transfer learning
- PyTorch training pipeline
- Dataset preprocessing
- Model persistence (.pth)
- Scalable project structure

## Dataset

- Dataset: PlantVillage
- Images: 17,675
- Classes: 10

### Classes

- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___Late_blight
- Tomato___Leaf_Mold
- Tomato___Septoria_leaf_spot
- Tomato___Spider_mites Two-spotted_spider_mite
- Tomato___Target_Spot
- Tomato___Tomato_Yellow_Leaf_Curl_Virus
- Tomato___Tomato_mosaic_virus
- Tomato___healthy

## Model

- Architecture: ResNet18
- Framework: PyTorch
- Transfer Learning: Yes

## Project Structure

```text
CropGuard
├── data
├── src
│   ├── data
│   ├── models
│   ├── training
│   └── utils
├── tests
├── cropguard_model.pth
└── README.md