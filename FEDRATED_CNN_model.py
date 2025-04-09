import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
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

def train_local(model, dataloader, device, local_epochs, optimizer, criterion):
    model.train()
    for epoch in range(local_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def average_weights(local_weights):
    avg_weights = copy.deepcopy(local_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
    return avg_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

client_dirs = {
    "client1": r"D:\KUSHAGRA\DATA\K_Works\SEAI Project\dataset\client1",
    "client2": r"D:\KUSHAGRA\DATA\K_Works\SEAI Project\dataset\client2",
    "client3": r"D:\KUSHAGRA\DATA\K_Works\SEAI Project\dataset\client3"
}

local_loaders = {}
batch_size = 16
for client, data_dir in client_dirs.items():
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    local_loaders[client] = loader

global_model = SimpleCNN().to(device)

num_rounds = 3
local_epochs = 2
lr = 0.001
criterion = nn.CrossEntropyLoss()

for rnd in range(num_rounds):
    print(f"Federated Learning Round {rnd+1}/{num_rounds}")
    local_weights = []
    for client, loader in local_loaders.items():
        print(f" Training on {client}'s data...")
        local_model = SimpleCNN().to(device)
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        w = train_local(local_model, loader, device, local_epochs, optimizer, criterion)
        local_weights.append(w)
    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)
    print(" Global model updated.\n")

global_model.eval()
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
        outputs = global_model(image_tensor)
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
