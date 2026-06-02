# gradcam.py
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import random
from PIL import Image
from torchvision import transforms, datasets

# Import models & config
from models.cnn import SimpleCNN
from models.cnn_cbam import SimpleCNN_CBAM
from models.cnn_se import SimpleCNN_SE
from config import *

# ---------------------------
# GradCAM Class
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, class_idx=None):
        output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1,2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        return cam.cpu().numpy()

# ---------------------------
# Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return image, tensor

# ---------------------------
# Overlay heatmap
# ---------------------------
def apply_colormap_on_image(org_img, cam, alpha=0.6):
    cam = cv2.resize(cam, (org_img.width, org_img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    heatmap = np.clip(heatmap*1.5, 0, 1)
    org_img = np.array(org_img).astype(np.float32) / 255
    overlay = heatmap * alpha + org_img
    overlay /= overlay.max()
    return np.uint8(255 * overlay)

# ---------------------------
# Generate Grad-CAM for a single model
# ---------------------------
def get_gradcam_image(model, target_layer, img_path, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device).eval()
    org_img, tensor = preprocess_image(img_path)
    tensor = tensor.to(device)
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(tensor)
    overlay = apply_colormap_on_image(org_img, cam)
    return overlay

# ---------------------------
# Add text labels
# ---------------------------
def add_labels(image, labels):
    h, w, _ = image.shape
    font_scale = 1
    thickness = 2
    label_img = image.copy()
    width_per_model = w // len(labels)
    for i, label in enumerate(labels):
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x = i * width_per_model + (width_per_model - text_size[0]) // 2
        y = 30
        cv2.putText(label_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return label_img

# ---------------------------
# Main routine
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "gradcam_results"
    os.makedirs(save_dir, exist_ok=True)

    # Load test dataset
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    # Select 10 random images per class
    class_indices = {cls: [] for cls in test_dataset.classes}
    for idx, (_, label) in enumerate(test_dataset.imgs):
        class_name = test_dataset.classes[label]
        class_indices[class_name].append(idx)

    selected_indices = []
    for cls in test_dataset.classes:
        selected_indices.extend(random.sample(class_indices[cls], min(10, len(class_indices[cls]))))

    # Model configurations
    models_info = [
        {"name": "CNN", "class": SimpleCNN, "weights": "CNN_model_best.pth", "layer": "conv2"},
        {"name": "CBAM", "class": SimpleCNN_CBAM, "weights": "CBAM_model_best.pth", "layer": "last_conv"},
        {"name": "SE", "class": SimpleCNN_SE, "weights": "SE_model_best.pth", "layer": "last_conv"},
    ]

    # Generate combined Grad-CAM images
    for i, idx in enumerate(selected_indices):
        img_path, label = test_dataset.imgs[idx]
        cls_name = test_dataset.classes[label]
        combined_images = []
        for info in models_info:
            model_class = info["class"]
            model_name = info["name"]
            model_path = info["weights"]
            layer_name = info["layer"]
            model = model_class(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
            target_layer = getattr(model, layer_name)
            overlay = get_gradcam_image(model, target_layer, img_path, model_path, device)
            combined_images.append(overlay)
        # Stack horizontally
        final_image = np.hstack(combined_images)
        # Add labels
        final_image = add_labels(final_image, [info["name"] for info in models_info])
        save_path = os.path.join(save_dir, f"{i+1}_{cls_name}_combined.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    print("✅ Combined Grad-CAM images with labels saved in:", save_dir)
