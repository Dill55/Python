import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning Parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # normalize images from dataset
])

train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True) # changing set
test_dataset = datasets.MNIST(root='data', train=False, transform=transform)                # control/compare set

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# neural network initialization
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Layers
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten image
        x = x.view(-1, 28*28)

        # Apply Layers reduction
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # output scores
        return x

# Initialize model and optimizer
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Send to GPU or CPU
        images, labels = images.to(device), labels.to(device)
        
        # Send to model
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)


        # Remove gradients 
        optimizer.zero_grad()
        # Adjustments
        loss.backward()
        # Learn loop
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # Send to GPU/CPU
        images, labels = images.to(device), labels.to(device)

        # Send to model
        outputs = model(images)

        # compare correctness vs total
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print total test accuracy
print(f'Test Accuracy: {100 * correct / total:.2f}%')