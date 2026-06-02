import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from models.cnn_cbam import SimpleCNN_CBAM
from models.cnn_se import SimpleCNN_SE
from config import *

# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Data
# -------------------------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -------------------------
# Logged values
# -------------------------
#CNN
# ---------------------------
# CNN logged values
# ---------------------------
cnn_train_loss = [0.4029,0.3400,0.3108,0.2809,0.2607,0.2344,0.2095,0.1884,0.1779,0.1597,
                  0.1533,0.1422,0.1311,0.1210,0.1255,0.1075,0.1123,0.1002,0.0895,0.0931,
                  0.0868,0.0811,0.0694,0.0794,0.0754,0.0735,0.0702]
cnn_train_acc = [85.19,86.75,87.73,89.27,90.01,90.93,92.19,92.92,93.47,94.13,
                 94.23,95.04,95.47,95.49,95.30,96.04,95.73,96.25,96.87,96.63,
                 96.89,96.96,97.18,97.15,96.96,97.30,97.32]
cnn_test_acc = [85.62,89.25,90.76,90.84,91.75,91.90,93.79,93.94,95.23,94.17,
                96.06,95.38,96.52,96.67,95.99,96.67,97.12,96.74,97.58,96.67,
                97.73,96.82,97.65,97.96,97.73,97.05,97.58]
epochs_cnn = range(4, 31)

# CBAM
cbam_train_loss = [0.1797,0.1482,0.1178,0.1006,0.0989,0.0751,0.0604,0.0622,0.0558,0.0512,
                   0.0424,0.0361,0.0341,0.0350,0.0244,0.0252,0.0177,0.0309,0.0237,0.0252,
                   0.0189,0.0136,0.0158,0.0207,0.0176]
cbam_train_acc  = [93.16,94.83,96.01,96.68,96.58,97.25,98.06,98.15,98.00,98.20,
                   98.69,98.91,98.79,98.77,99.17,99.10,99.52,99.03,99.31,99.31,
                   99.38,99.53,99.50,99.46,99.53]
cbam_test_acc   = [92.96,94.32,95.69,96.97,96.52,97.27,95.99,98.03,96.97,96.14,
                   98.03,98.49,97.73,98.49,97.35,97.73,98.56,98.33,98.71,98.03,
                   98.71,98.64,98.03,98.64,98.64]
epochs_cbam = range(6, 31)

# SE
se_train_loss = [0.2633,0.2145,0.1722,0.1427,0.1217,0.0983,0.0834,0.0846,0.0593,0.0540,
                 0.0480,0.0444,0.0393,0.0337,0.0408,0.0273,0.0302,0.0250,0.0279,0.0232,
                 0.0206,0.0187,0.0175,0.0148,0.0173,0.0151,0.0193]
se_train_acc  = [90.12,91.81,93.83,95.15,95.59,96.16,97.03,97.01,98.00,98.27,
                 98.58,98.45,98.70,98.86,98.77,99.17,99.21,99.05,99.12,99.29,
                 99.31,99.43,99.46,99.57,99.46,99.55,99.41]
se_test_acc   = [89.17,92.51,93.41,94.17,96.21,96.14,96.44,95.91,97.96,97.20,
                 97.27,97.43,98.79,97.50,98.56,97.43,98.11,98.18,98.79,98.18,
                 97.50,98.49,98.41,97.50,98.56,98.79,98.56]
epochs_se = range(4, 31)

# -------------------------
# Function to compute ROC
# -------------------------
def get_roc(model_path, model_class, model_name):
    model = model_class(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_outputs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = F.softmax(model(images), dim=1)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels)
    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Binarize for multiclass ROC
    y_true = label_binarize(all_labels, classes=range(NUM_CLASSES))
    roc_dict = {}
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true[:, i], all_outputs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_dict[i] = (fpr, tpr, roc_auc)
    return roc_dict

# -------------------------
# CBAM ROC
# -------------------------
cbam_roc = get_roc("CBAM_model_best.pth", SimpleCNN_CBAM, "CBAM")

# -------------------------
# SE ROC
# -------------------------
se_roc = get_roc("SE_model_best.pth", SimpleCNN_SE, "SE")

# ---------------------------
# Plot CNN Accuracy & Loss
# ---------------------------
plt.figure(figsize=(12,6))
plt.plot(epochs_cnn, cnn_train_acc, '--', label="CNN Train Acc")
plt.plot(epochs_cnn, cnn_test_acc, label="CNN Test Acc")
plt.plot(epochs_cnn, cnn_train_loss, label="CNN Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("CNN Accuracy & Loss")
plt.legend()
plt.grid(True)
plt.savefig("cnn_results.png", dpi=300)
plt.close()

# -------------------------
# Plot merged CBAM
# -------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(epochs_cbam, cbam_train_acc, '--', label="Train Acc")
plt.plot(epochs_cbam, cbam_test_acc, label="Test Acc")
plt.plot(epochs_cbam, cbam_train_loss, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("CBAM Accuracy & Loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
for cls, (fpr, tpr, roc_auc) in cbam_roc.items():
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC={roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("CBAM ROC Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cbam_merged.png", dpi=300)
plt.close()

# -------------------------
# Plot merged SE
# -------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(epochs_se, se_train_acc, '--', label="Train Acc")
plt.plot(epochs_se, se_test_acc, label="Test Acc")
plt.plot(epochs_se, se_train_loss, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("SE Accuracy & Loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
for cls, (fpr, tpr, roc_auc) in se_roc.items():
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC={roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("SE ROC Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("se_merged.png", dpi=300)
plt.close()

print("✅ Saved: cbam_merged.png, se_merged.png")
