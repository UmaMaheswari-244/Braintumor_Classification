# metrics_plot.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

from models.cnn import SimpleCNN
from models.cnn_cbam import SimpleCNN_CBAM
from models.cnn_se import SimpleCNN_SE
from config import *

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Data transforms
# -------------------------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = test_dataset.classes

# -------------------------
# Helper: compute metrics
# -------------------------
def get_metrics(model_class, model_weights_path):
    model = model_class(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    # Load saved state dict
    model.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = F.softmax(model(images), dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    return precision, recall, f1

# -------------------------
# Models info
# -------------------------
models_info = [
    {"name": "CNN", "class": SimpleCNN, "weights": MODEL_SAVE_PATH_CNN},
    {"name": "CNN+CBAM", "class": SimpleCNN_CBAM, "weights": MODEL_SAVE_PATH_CBAM},
    {"name": "CNN+SE", "class": SimpleCNN_SE, "weights": MODEL_SAVE_PATH_SE},
]

# -------------------------
# Compute metrics
# -------------------------
all_metrics = {}
for info in models_info:
    precision, recall, f1 = get_metrics(info["class"], info["weights"])
    all_metrics[info["name"]] = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -------------------------
# Plot metrics
# -------------------------
x = np.arange(NUM_CLASSES)
width = 0.2

plt.figure(figsize=(12,6))

for i, metric_name in enumerate(["precision", "recall", "f1"]):
    plt.subplot(1,3,i+1)
    for j, info in enumerate(models_info):
        values = all_metrics[info["name"]][metric_name]
        plt.bar(x + j*width, values, width=width, label=info["name"])
    plt.xticks(x + width, classes, rotation=45)
    plt.ylim(0,1)
    plt.title(metric_name.upper())
    plt.ylabel(metric_name)
    plt.grid(axis='y')
    if i==0:
        plt.legend()

plt.tight_layout()
plt.savefig("precision_recall_f1_comparison.png", dpi=300)
plt.show()
print("✅ Saved precision_recall_f1_comparison.png")
