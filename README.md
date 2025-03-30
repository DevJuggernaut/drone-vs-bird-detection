# Drone vs Bird Classification using Computer Vision

A computer vision project for accurate classification of drone and bird images using PyTorch and transfer learning based on the ResNet-18 architecture.

## Project Description

This project demonstrates the development of a deep learning model to distinguish between drones and birds in images. Such technology can be applied in airport security systems, airspace management, and wildlife monitoring.

### Features
- Transfer learning with ResNet-18 architecture
- Achieving 99.39% accuracy on the test set
- Complete development cycle: from data processing to model deployment

## Dataset

The project uses the "Drone vs Bird" dataset from Kaggle, which contains:
- 2499 drone images
- 1607 bird images
- Various backgrounds and shooting conditions

## Results

- **Accuracy**: 99.39%
- **F1-score**: 0.9939
- **AUC**: 0.9993
- **Confusion Matrix**:
  - 322/322 birds classified correctly
  - 495/500 drones classified correctly

## Project Structure

This project is organized as a single Python notebook containing all the steps:

1. **Environment Setup & Data Loading**: Installing required libraries and downloading data
2. **Data Exploration**: Analyzing dataset structure and image characteristics
3. **Data Preparation**: Creating datasets with transformations and splitting into train/val/test sets
4. **Model Creation & Training**: Implementing transfer learning with ResNet-18
5. **Evaluation**: Testing the model and visualizing results

## Required Libraries

Main libraries:
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Kaggle Hub

## Installation and Usage

1. Clone the repository:
```
git clone https://github.com/your-username/drone-vs-bird-detection.git
cd drone-vs-bird-detection
```

2. Install the required dependencies:
```
pip install torch torchvision opencv-python matplotlib scikit-learn kagglehub
```

3. Run the Jupyter notebook to see the full project:
```
jupyter notebook drone_vs_bird_detection.ipynb
```

4. To use the trained model for inference:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the model architecture
class DroneVsBirdModel(nn.Module):
    def __init__(self):
        super(DroneVsBirdModel, self).__init__()
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.model(x)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DroneVsBirdModel()
model.load_state_dict(torch.load('drone_vs_bird_model.pth', map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classification function
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    class_names = {0: "Bird", 1: "Drone"}
    return class_names[predicted.item()], probabilities[0].cpu().numpy()

# Example usage
image_path = "your_image.jpg"
prediction, probs = classify_image(image_path)
print(f"Prediction: {prediction}")
print(f"Probability - Bird: {probs[0]:.4f}, Drone: {probs[1]:.4f}")
```
