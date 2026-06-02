import os
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms, datasets
import shap
import matplotlib.pyplot as plt

# Import models & config
# NOTE: Ensure models/cnn.py, models/cnn_cbam.py, models/cnn_se.py,
# and config.py are accessible, and SimpleCNN is defined in simple_cnn_train.py
from simple_cnn_train import SimpleCNN 
from models.cnn_cbam import SimpleCNN_CBAM
from models.cnn_se import SimpleCNN_SE
from config import * # IMG_SIZE, BATCH_SIZE, NUM_CLASSES, TEST_DIR, etc.

# --- 1. Utility: Data Preprocessing ---
# Unnormalized transform for image plotting (SHAP plots unnormalized images)
plot_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), # converts to [0, 1] range
])

# Normalized transform for model prediction
model_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image_data(img_path):
    """Loads image, converts to PIL, tensor (for model), and tensor (for plot)."""
    image = Image.open(img_path).convert("RGB")
    tensor_model = model_transform(image).unsqueeze(0) # For model input (normalized)
    tensor_plot = plot_transform(image).unsqueeze(0)  # For SHAP plot background (unnormalized, [0, 1])
    return tensor_model, tensor_plot

# --- 2. SHAP Calculation and Plotting ---

def get_shap_image(model, img_path, model_path, background, device, class_names):
    """Loads model, calculates SHAP values, and returns the SHAP visualization array."""
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device).eval()
    
    # Prepare image data
    tensor_model, tensor_plot = load_image_data(img_path)
    tensor_model = tensor_model.to(device)
    
    # ----------------------------------------------------------------------
    # SHAP Logic
    # ----------------------------------------------------------------------
    # 1. Define explainer (DeepExplainer/GradientExplainer is best for CNNs)
    # The DeepExplainer requires a background set for the baseline.
    explainer = shap.DeepExplainer(model, background.to(device))
    
    # 2. Calculate SHAP values
    shap_values = explainer.shap_values(tensor_model)
    
    # 3. Get predicted class for labeling
    with torch.no_grad():
        output = model(tensor_model)
        pred_class_idx = output.argmax(dim=1).item()
        pred_class = class_names[pred_class_idx]
        
    # 4. Prepare data for plotting (convert to NumPy and reorder dimensions)
    # Pytorch: (N, C, H, W) -> SHAP plot needs: (N, H, W, C)
    shap_numpy = [np.transpose(s, (0, 2, 3, 1)) for s in shap_values]
    test_numpy = np.transpose(tensor_plot.cpu().numpy(), (0, 2, 3, 1))
    
    # 5. Generate the actual SHAP visualization *without* showing it
    # We use the internal matplotlib figure of shap.image_plot to save it as an array
    # We are generating a figure just to get the pixel data of the resulting plot
    
    # Get SHAP values for the predicted class
    shap_for_predicted_class = shap_numpy[pred_class_idx][0]
    
    # Create the SHAP image overlay (SHAP does not have an easy return array)
    # We'll use a standard Matplotlib-based approach to capture the output image.
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # Use the custom SHAP plotting function logic to get the overlay for the predicted class
    # The raw SHAP output for one image is the final plot, which is complex to extract
    # as a simple numpy array. We will generate the SHAP image plot and capture it.
    
    # NOTE: SHAP plots the raw pixel values (0-1) in the background.
    # The official shap.image_plot is designed for display, not array return.
    
    # For a simple SHAP overlay equivalent:
    # We will manually create the overlay using the SHAP map and the original image.
    
    # Convert SHAP map to an overlay
    cam = shap_for_predicted_class[:, :, 0] # Use only one channel for simplicity
    cam = (cam - cam.min()) / (cam.max() - cam.min()) if cam.max() != cam.min() else cam
    cam = np.clip(cam, 0, 1)
    
    # Resize and apply heatmap (reusing the logic from your Grad-CAM)
    org_img_pil = Image.open(img_path).convert("RGB")
    cam_resized = cv2.resize(cam, (org_img_pil.width, org_img_pil.height), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    alpha = 0.6
    heatmap = np.float32(heatmap) / 255
    heatmap = np.clip(heatmap*1.5, 0, 1) 
    org_img_np = np.array(org_img_pil).astype(np.float32) / 255
    overlay_final = heatmap * alpha + org_img_np * (1 - alpha)
    overlay_final = np.uint8(255 * (overlay_final / overlay_final.max()))

    return overlay_final, pred_class

# --- 3. Labeling and Main Routine (Copied from gradcam.py) ---
def add_labels(image, labels):
    # ... (Keep your existing add_labels function) ...
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "shap_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test dataset for background and image paths
    # Note: Use the plot_transform just to get the path/label data
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=plot_transform)
    class_names = test_dataset.classes

    # --- Prepare Background Data for SHAP ---
    # SHAP DeepExplainer needs a small set of background samples (e.g., 10-50 images)
    background_tensors = []
    # Use the full data loader to get a batch
    bg_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(TEST_DIR, transform=model_transform), # Use normalized transform
        batch_size=BATCH_SIZE
    )
    for i, (images, _) in enumerate(bg_loader):
        background_tensors.append(images)
        if i * BATCH_SIZE >= 32: # Use 32 images as a good baseline
            break
    background = torch.cat(background_tensors)
    print(f"Prepared SHAP background set with {background.shape[0]} images.")
    # ----------------------------------------

    # Select 10 random images per class
    class_indices = {cls: [] for cls in test_dataset.classes}
    for idx, (_, label) in enumerate(test_dataset.imgs):
        class_name = test_dataset.classes[label]
        class_indices[class_name].append(idx)

    selected_indices = []
    for cls in test_dataset.classes:
        # Select 10 random images per class (or min available)
        selected_indices.extend(random.sample(class_indices[cls], min(10, len(class_indices[cls]))))

    # Model configurations (using the paths from your config)
    models_info = [
        {"name": "CNN", "class": SimpleCNN, "weights": MODEL_SAVE_PATH_CNN},
        {"name": "CBAM", "class": SimpleCNN_CBAM, "weights": MODEL_SAVE_PATH_CBAM},
        {"name": "SE", "class": SimpleCNN_SE, "weights": MODEL_SAVE_PATH_SE},
    ]

    # Generate combined SHAP images
    print("Starting SHAP visualization process...")
    for i, idx in enumerate(selected_indices):
        img_path, true_label_idx = test_dataset.imgs[idx]
        cls_name = test_dataset.classes[true_label_idx]
        combined_images = []
        
        for info in models_info:
            model_class = info["class"]
            model_name = info["name"]
            model_path = info["weights"]
            
            # 1. Instantiate a new model instance
            model = model_class(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            
            # 2. Get the SHAP-generated image overlay
            try:
                overlay, pred_class = get_shap_image(model, img_path, model_path, background, device, class_names)
                combined_images.append(overlay)
                print(f"   [SHAP-{model_name}] Image {i+1} done. True: {cls_name}, Pred: {pred_class}")
            except Exception as e:
                print(f"   [SHAP-{model_name}] ERROR processing image {i+1}: {e}")
                # Append a black image placeholder on error
                combined_images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)) 
                
        # Resize all images to the same height (in case of subtle size differences)
        target_h = combined_images[0].shape[0]
        resized_images = [cv2.resize(img, (img.shape[1], target_h)) for img in combined_images]

        # Stack horizontally
        final_image = np.hstack(resized_images)
        
        # Add labels
        final_image = add_labels(final_image, [f'{info["name"]}' for info in models_info])
        
        save_path = os.path.join(save_dir, f"{i+1}_{cls_name}_SHAP_combined.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    print("\n✅ Combined SHAP images with labels saved in:", save_dir)