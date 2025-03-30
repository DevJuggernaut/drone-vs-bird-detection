# Drone vs Bird Classification using Computer Vision

A computer vision project for accurate classification of drone and bird images using PyTorch and transfer learning based on the ResNet-18 architecture.

## Project Description

This project demonstrates the development of a deep learning model to distinguish between drones and birds in images. Such technology can be applied in airport security systems, airspace management, and wildlife monitoring.

### Features
- Transfer learning with ResNet-18 architecture
- Achieving 99.15% accuracy on the test set
- Complete development cycle: from data processing to model deployment

## Dataset

The project uses the "Drone vs Bird" dataset from Kaggle, which contains:
- 2499 drone images
- 1607 bird images
- Total of 4106 images with various backgrounds and shooting conditions

Drone images are generally larger (ranging from 752×480 to 3024×4032 pixels) compared to bird images (typically 224×225 to 500×500 pixels).

## Data Preparation

The data was split into:
- Training set: 2627 images (1599 drones, 1028 birds)
- Validation set: 657 images (400 drones, 257 birds)
- Test set: 822 images (500 drones, 322 birds)

All images were resized to 224×224 pixels and augmented with techniques such as random horizontal flips, rotations, and color jitter to improve model generalization.

## Model Architecture

- Base model: ResNet-18 (pretrained on ImageNet)
- Modified with custom classification head:
  - Linear layer (512 neurons)
  - ReLU activation
  - Dropout (0.2)
  - Output layer (2 classes)
- Training approach: Freeze early layers, fine-tune later layers

## Results

- **Accuracy**: 99.15%
- **Precision**: 98.98%
- **Recall**: 99.24%
- **F1-score**: 99.11%
- **AUC**: 0.9995
- **Confusion Matrix**:
  - 321/322 birds classified correctly (99.69% recall)
  - 494/500 drones classified correctly (98.80% recall)
  - Only 7 misclassifications out of 822 test images

## Model Download

Since the trained model file (45 MB) exceeds GitHub's file size limit (25 MB), you can download it from:

- **Google Drive**: [Download drone_vs_bird_model.pth](https://drive.google.com/file/d/1TGsCjLdJeXdiPgViaTr0r5Wg1s4t8O71/view?usp=sharing)

To use the model, download the file and place it in the project root directory before running the inference code.

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

3. Download the model from the Google Drive link above

4. Run the Jupyter notebook to see the full project:
```
jupyter notebook drone_vs_bird_detection.ipynb
```

5. To use the trained model for inference:

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
image_path = "path_to_your_image.jpg"
prediction, probs = classify_image(image_path)
print(f"Prediction: {prediction}")
print(f"Probability - Bird: {probs[0]:.4f}, Drone: {probs[1]:.4f}")
```

## Alternative: Training the Model Yourself

If you prefer to train the model yourself:

1. Run the complete notebook which will download the dataset and train the model
2. The trained model will be saved as 'drone_vs_bird_model.pth' in your working directory
3. You can then use this file for inference using the code provided above
