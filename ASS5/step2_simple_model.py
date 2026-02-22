import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
import random
from tqdm import tqdm 
from dataset_module import PotsdamDataset


class SimpleSegmenter(nn.Module):
    def __init__(self):
        super(SimpleSegmenter, self).__init__()
        # Input layer: 4 channels (RGB + IR), Output: 32 filters
        # kernel_size=3, padding=1 (equivalent to 'same' in TF) 
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1) 
        self.relu = nn.ReLU()
        
        # Output layer: 32 channels in, 6 channels out (for 6 classes)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=6, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # We output raw logits here. nn.CrossEntropyLoss handles the Softmax calculation 
        x = self.conv2(x) 
        return x
    
if __name__ == '__main__':
    # Load data splits from step 1
    with open('data_splits.json', 'r') as f:
        splits = json.load(f)

    # For step 2, input_bands='rgb_ir'
    train_dataset = PotsdamDataset(splits['train'], input_bands='rgb_ir')
    val_dataset = PotsdamDataset(splits['val'], input_bands='rgb_ir')
    test_dataset = PotsdamDataset(splits['test'], input_bands='rgb_ir')

    # num_workers = 0 because bizarre bug with python 3.14 on Windows when combined with pytorch multiprocessing.
    # if on Linux, we can use num_workers = 4 or 8 to take full advantages of my RTX 4080
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda")
    model = SimpleSegmenter().to(device)

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

            # Perform lightning-fast augmentation on GPU
            # Random Horizontal Flip
            if random.random() > 0.5:
                inputs = torch.flip(inputs, [3]) 
                labels = torch.flip(labels, [2]) 
            
            # Random Vertical Flip
            if random.random() > 0.5:
                inputs = torch.flip(inputs, [2]) 
                labels = torch.flip(labels, [1]) 
                
            # Random Rotation: 0, 90, 180, 270
            k = random.randint(0, 3)
            if k > 0:
                inputs = torch.rot90(inputs, k, [2, 3]) 
                labels = torch.rot90(labels, k, [1, 2])

            # Deep Learning Forward/Backward Pass
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
            torch.save(best_model_wts, 'best_simple_model.pth')
            print(">> Best model saved!")

    # Evaluate on test set
    model.load_state_dict(torch.load('best_simple_model.pth'))
    model.eval()
    test_loss, test_corrects, test_pixels = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
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
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()