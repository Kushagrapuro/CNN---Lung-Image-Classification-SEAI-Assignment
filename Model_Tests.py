'''
Image: D:\KUSHAGRA\DATA\K_Works\SEAI Project\dataset\test\COVID\COVID-1.png
Predicted Class: 0 (COVID)
Class Probabilities: [[9.9989140e-01 1.0859349e-04]]

Image: D:\KUSHAGRA\DATA\K_Works\SEAI Project\dataset\test\COVID\COVID-3.png
Predicted Class: 0 (COVID)
Class Probabilities: [[9.9999988e-01 7.5389295e-08]]

'''

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


model_path = r"D:\KUSHAGRA\DATA\K_Works\SEAI Project\Trained Model\best_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


test_image_paths = [
    r"D:\KUSHAGRA\DATA\K_Works\SEAI Project\dataset\test\COVID\COVID-1.png",  
    r"D:\KUSHAGRA\DATA\K_Works\SEAI Project\dataset\test\COVID\COVID-3.png"  
]
true_labels = [0, 1]  


predicted_labels = []
predicted_scores = []

for img_path in test_image_paths:
    
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  

    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    
    predicted_labels.append(predicted.item())
    predicted_scores.append(probabilities.cpu().numpy())

    
    print(f"Image: {img_path}")
    print(f"Predicted Class: {predicted.item()} ({'COVID' if predicted.item() == 0 else 'Viral Pneumonia'})")
    print(f"Class Probabilities: {probabilities.cpu().numpy()}\n")


conf_matrix = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["COVID", "Viral Pneumonia"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()