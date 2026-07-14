import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# MODEL DEFINITION
class SmallCNN(nn.Module):
    """A small Convolutional Neural Network for Fashion-MNIST."""
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# DATA LOADING FUNCTION
def get_dataloaders(batch_size=64):
    """Downloads Fashion-MNIST and returns DataLoaders."""
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return train_loader, test_loader, class_names

# TRAINING FUNCTION
def train_model(model, train_loader, num_epochs=5, lr=0.001):
    """Trains the model and returns the optimizer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on device: {device}\n")
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    return optimizer

# EVALUATION FUNCTION
def evaluate_model(model, test_loader):
    """Evaluates the model on the test set and returns accuracy."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nAccuracy on the test set: {accuracy:.2f}%')
    return accuracy

# VISUALIZATION FUNCTION

def visualize_predictions(model, test_loader, class_names, num_images=6):
    """Displays a grid of images with true vs predicted labels."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(12, 5))
    for idx in np.arange(num_images):
        ax = fig.add_subplot(1, num_images, idx+1, xticks=[], yticks=[])
        img = images[idx].cpu() * 0.5 + 0.5  # Unnormalize
        plt.imshow(img.squeeze(), cmap='gray')

        color = "green" if preds[idx] == labels[idx] else "red"
        ax.set_title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}", color=color)

    plt.tight_layout()
    plt.show()
