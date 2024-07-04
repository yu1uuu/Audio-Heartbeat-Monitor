import os
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)
!ls "/content/gdrive/My Drive/aps360/feedforward"

base_path = "/content/drive/MyDrive/aps360/feedforward/"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Feedforward_HeartbeatClassifier(nn.Module):
    def __init__(self):
        super(Feedforward_HeartbeatClassifier, self).__init__()
        self.fc1 = nn.Linear(3 * 128 * 128, 512)  # input size is 3*128*128 (3 channels, 128x128 pixels)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 3 * 128 * 128)  # flatten the image input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))  # using sigmoid for binary classification
        return x

model = Feedforward_HeartbeatClassifier()

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # resize to 128x128 pixels
    transforms.ToTensor()  
])

train_dir = '/content/gdrive/My Drive/aps360/feedforward/Train'
val_dir = '/content/gdrive/My Drive/aps360/feedforward/Val'

train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(train_dataset.classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_accuracy(outputs, labels):
    preds = outputs.round()
    correct = preds.eq(labels.view_as(preds)).sum().item()
    return correct / len(labels)

def plot(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))

    # loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def train(model, train_loader, val_loader, num_epochs=4, batch_size=64, learn_rate=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Training Starting...")
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            correct_train += calculate_accuracy(outputs, labels.unsqueeze(1).float()) * labels.size(0)
            total_train += labels.size(0)
        
        train_loss = training_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()
                correct_val += calculate_accuracy(outputs, labels.unsqueeze(1).float()) * labels.size(0)
                total_val += labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    print('Finished Training')
    plot(train_losses, val_losses, train_accuracies, val_accuracies)
    model_path = '/content/gdrive/My Drive/aps360/feedforward/feedforward_heartbeat_classifier.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

train(model, train_loader, val_loader, num_epochs=4, batch_size=64, learn_rate=0.001)
