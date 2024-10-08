# -*- coding: utf-8 -*-
"""Copy of saturday - cnn model 3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sVzpd22xGFzbRyqePy_SCCsVIkS_eBgs

Colab link: https://colab.research.google.com/drive/1aQFUTW2nmGHkJ0HlAz9qWAlKuIvejNY-?usp=sharing
"""

from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)
!ls "/content/gdrive/My Drive/APS360/feedforward"

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class CNN_HeartbeatClassifier(nn.Module):
    def __init__(self):
        super(CNN_HeartbeatClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.dropout1 = nn.Dropout(0.35)
        self.dropout2 = nn.Dropout(0.55)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout1(self.pool(self.bn2(F.relu(self.conv2(x)))))
        x = self.dropout1(self.pool(F.relu(self.conv3(x))))
        x = x.view(-1, 128 * 14 * 14)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # resize to 128x128 pixels
    transforms.ToTensor()
])

train_dir = '/content/gdrive/My Drive/APS360/feedforward/Train'
val_dir = '/content/gdrive/My Drive/APS360/feedforward/Val'
test_dir = '/content/gdrive/My Drive/APS360/feedforward/Test'
final_test_dir = '/content/gdrive/My Drive/APS360/feedforward/Test (New)'

train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)
final_test_dataset = ImageFolder(root=final_test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
final_test_loader = DataLoader(final_test_dataset, batch_size=32, shuffle=False)

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

def train(model, train_loader, val_loader, num_epochs=4, batch_size=64, learn_rate=0.001, weight_decay=1e-5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
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
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
    model_path = '/content/gdrive/My Drive/APS360/cnn_heartbeat_classifier.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def plot_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.round().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # metrics
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def test_results(model, test_loader):
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No need to compute gradients
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    print("Test Accuracy: {:.2f}%".format(accuracy * 100))
    print("Test Precision: {:.2f}%".format(precision * 100))
    print("Test Recall: {:.2f}%".format(recall * 100))
    print("Test F1 Score: {:.2f}%".format(f1 * 100))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

model = CNN_HeartbeatClassifier()

train(model, train_loader, val_loader, num_epochs=33, batch_size=32, learn_rate=0.0005, weight_decay=1e-5)

plot_confusion_matrix(model, val_loader, device)

test_results(model, test_loader)

plot_confusion_matrix(model, test_loader, device)

test_results(model, final_test_loader)

plot_confusion_matrix(model, final_test_loader, device)
