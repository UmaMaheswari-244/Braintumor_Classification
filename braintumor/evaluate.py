import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import models
from models.cnn import SimpleCNN          # Plain CNN
from models.cnn_cbam import SimpleCNN_CBAM
from models.cnn_se import SimpleCNN_SE
from config import *  # Should contain IMG_SIZE, BATCH_SIZE, NUM_CLASSES, TEST_DIR

# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Data transform
# -------------------------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Evaluation function
# -------------------------
def evaluate_model(model, model_name, model_path):
    # Load weights with strict=False to avoid mismatch errors
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except Exception as e:
        print(f"⚠️ Warning loading {model_name}: {e}")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    total = len(all_labels)
    accuracy = correct / total * 100
    print(f"\n📊 {model_name} Test Accuracy: {accuracy:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    return accuracy

# -------------------------
# Evaluate all models
# -------------------------
cnn_model = SimpleCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
cbam_model = SimpleCNN_CBAM(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
se_model = SimpleCNN_SE(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)

cnn_acc = evaluate_model(cnn_model, "CNN", "CNN_model_best.pth")
cbam_acc = evaluate_model(cbam_model, "CBAM", "CBAM_model_best.pth")
se_acc = evaluate_model(se_model, "SE", "SE_model_best.pth")

# -------------------------
# Summary
# -------------------------
print("\n================ SUMMARY ================")
print(f"CNN Accuracy   = {cnn_acc:.2f}%")
print(f"CBAM Accuracy  = {cbam_acc:.2f}%")
print(f"SE Accuracy    = {se_acc:.2f}%")

best_model = "CNN"
if cbam_acc > cnn_acc and cbam_acc > se_acc:
    best_model = "CBAM"
elif se_acc > cnn_acc and se_acc > cbam_acc:
    best_model = "SE"

print(f"✅ Best Model = {best_model}")
print("==========================================")
