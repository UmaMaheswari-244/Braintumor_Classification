import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.cnn_cbam import SimpleCNN_CBAM
from models.cnn_se import SimpleCNN_SE   # <-- add SE version
from config import *

# Data transforms with augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation/test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Training Function (shared for both models) ----------
def train_and_evaluate(model, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total * 100

        # Evaluate on test set
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = test_correct / test_total * 100
        print(f"[{model_name}] Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # Save trained model
    save_path = f"{model_name}_{MODEL_SAVE_PATH}"
    torch.save(model.state_dict(), save_path)
    print(f"{model_name} model saved at {save_path}\n")


# ---------- Run Both Models ----------
print("🚀 Training CNN + CBAM...")
model_cbam = SimpleCNN_CBAM(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
train_and_evaluate(model_cbam, "CBAM")

print("🚀 Training CNN + SE...")
model_se = SimpleCNN_SE(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
train_and_evaluate(model_se, "SE")
