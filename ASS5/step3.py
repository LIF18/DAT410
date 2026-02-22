import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
import random
import numpy as np
from tqdm import tqdm
from dataset_module import PotsdamDataset

# 1. Define the Encoder-Decoder (U-Net) Architecture
class EncoderDecoderSegmenter(nn.Module):
    def __init__(self):
        super(EncoderDecoderSegmenter, self).__init__()
        
        # --- ENCODER ---
        # Input: (224, 224, 5) -> Conv (Filters=32, 3x3) -> (224, 224, 32)
        self.enc_conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (112, 112, 32)
        
        # Conv (Filters=64, 3x3) -> (112, 112, 64)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (56, 56, 64)
        
        # Bottleneck Conv (Filters=64, 3x3) -> (56, 56, 64)
        self.bottleneck = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # --- DECODER ---
        # First ConvTranspose (Filters=64, kernel=3, stride=2) (56, 56, 64) -> (112, 112, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Second ConvTranspose (After Concat 128): (112, 112, 128) -> (224, 224, 32) 
        self.upconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Third ConvTranspose (After Concat 64): (224, 224, 64) -> (224, 224, 32)
        # Stride is 1 here because spatial dimensions don't change, acting as a feature reducer.
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)

        # Final Output Convolution: (224, 224, 32) -> (224, 224, 6)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=6, kernel_size=3, padding=1)

    def forward(self, x):
            # Encoder
            e1 = self.relu(self.enc_conv1(x))  # (224, 224, 32)
            p1 = self.pool1(e1)                # (112, 112, 32)
            
            e2 = self.relu(self.enc_conv2(p1))  # (112, 112, 64)
            p2 = self.pool2(e2)                # (56, 56, 64)
            
            b = self.relu(self.bottleneck(p2))  # (56, 56, 64)
            
            # Decoder
            d1 = self.relu(self.upconv1(b))    # (112, 112, 64)
            d1 = torch.cat((d1, e2), dim=1)    # (112, 112, 128)
            
            d2 = self.relu(self.upconv2(d1))   # (224, 224, 32)
            d2 = torch.cat((d2, e1), dim=1)    # (224, 224, 64)
            
            d3 = self.relu(self.upconv3(d2))   # (224, 224, 32)
            out = self.final_conv(d3)          # (224, 224, 6) (Raw logits)
            
            return out

if __name__ == '__main__':
    # load data splits from step 1
    with open('data_splits.json', 'r') as f:
        splits = json.load(f)

    # For step 3, input_bands='all'
    train_dataset = PotsdamDataset(splits['train'], input_bands='all')
    val_dataset = PotsdamDataset(splits['val'], input_bands='all')
    test_dataset = PotsdamDataset(splits['test'], input_bands='all')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda")
    model = EncoderDecoderSegmenter().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Training on device: {device}")

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, running_corrects, total_pixels = 0.0, 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # GPU Data Augmentation
            if random.random() > 0.5:
                inputs, labels = torch.flip(inputs, [3]), torch.flip(labels, [2])
            if random.random() > 0.5:
                inputs, labels = torch.flip(inputs, [2]), torch.flip(labels, [1])
            k = random.randint(0, 3)
            if k > 0:
                inputs, labels = torch.rot90(inputs, k, [2, 3]), torch.rot90(labels, k, [1, 2])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_pixels += labels.numel()
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = running_corrects.double() / total_pixels
        
        # Validation
        model.eval()
        val_loss, val_corrects, val_pixels = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_pixels += labels.numel()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_corrects.double() / val_pixels
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())
        
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'best_unet_model.pth')
            print(">> Best U-Net model saved!")

    # Evaluation & Visualization on test set
    model.load_state_dict(torch.load('best_unet_model.pth'))
    model.eval()
    test_loss, test_corrects, test_pixels = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="[Test]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            test_pixels += labels.numel()

    final_test_loss = test_loss / len(test_dataset)
    final_test_acc = test_corrects.double() / test_pixels
    print(f"\nFinal Test Results -> Loss: {final_test_loss:.4f}, Accuracy: {final_test_acc:.4f}")

    # 5. Plotting curves 
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('U-Net Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('U-Net Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # plt.savefig('step 3 curves.png')
    plt.show()


    # Take ONE batch from the test set for the required visualization
    test_iter = iter(test_loader)
    test_inputs, test_labels = next(test_iter)
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    
    with torch.no_grad():
        test_preds = model(test_inputs)
        _, test_preds_classes = torch.max(test_preds, 1)
        
    # Move exactly one sample to CPU for matplotlib plotting
    sample_idx = 0
    img_tensor = test_inputs[sample_idx].cpu().numpy()
    target_map = test_labels[sample_idx].cpu().numpy()
    pred_map = test_preds_classes[sample_idx].cpu().numpy()

    # Reconstruct RGB (Indices 0, 1, 2) and Normalize for display
    rgb_img = img_tensor[0:3, :, :].transpose(1, 2, 0)
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img) + 1e-8)
    # Elevation is index 4
    elevation_img = img_tensor[4, :, :]

    # 5. Plotting Requirement
    class_names = [
        'Impervious surface', 
        'Building', 
        'Tree', 
        'Low vegetation', 
        'Car', 
        'Clutter/Background'
    ]
    colors = ['#FFFFFF', '#0000FF', '#00FF00', '#00FFFF', '#FFFF00', '#FF0000']
    cmap_custom = ListedColormap(colors)
    tick_positions = [0.4, 1.25, 2.1, 2.9, 3.75, 4.6]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(rgb_img)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    im_elev = axes[1].imshow(elevation_img, cmap='terrain')
    axes[1].set_title('Elevation Band')
    axes[1].axis('off')
    fig.colorbar(im_elev, ax=axes[1], fraction=0.046, pad=0.04, label='Elevation')

    im_target = axes[2].imshow(target_map, cmap=cmap_custom, vmin=0, vmax=5)
    axes[2].set_title('Target Labels')
    axes[2].axis('off')
    cbar1 = fig.colorbar(im_target, ax=axes[2], ticks=tick_positions, fraction=0.046, pad=0.04)
    cbar1.ax.set_yticklabels(class_names)

    im_pred = axes[3].imshow(pred_map, cmap=cmap_custom, vmin=0, vmax=5)
    axes[3].set_title('Model Prediction')
    axes[3].axis('off')
    cbar2 = fig.colorbar(im_pred, ax=axes[3], ticks=tick_positions, fraction=0.046, pad=0.04)
    cbar2.ax.set_yticklabels(class_names)

    plt.tight_layout()
    plt.show()