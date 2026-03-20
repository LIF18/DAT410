import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from dataset import GomokuFastDataset
from model import GomokuResNet


def set_seed(seed=42):

    # Sets the random seed for completely reproducible results.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to {seed} for reproducibility.")


def plot_training_history(train_losses, val_losses, train_accs, val_accs):

    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (NLL)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("\nTraining curves saved as 'training_curves.png'")


def train_model():

    set_seed(42)
    # Hyperparameter Settings (Optimized for RTX 4080)
    pt_filepath = "gomoku_500k.pt"
    batch_size = 1024  # RTX 4080 has large VRAM, 1024 greatly accelerates training
    epochs = 20        # Run for 20 Epochs (Ref: Assignment 5)
    learning_rate = 1e-3
    validation_split = 0.1 # 10% of data used for validation
    save_path = "best_resnet_gomoku.pth" 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current computation device: {device}")
    if device.type == 'cpu':
        print("WARNING: No GPU detected. Training will be extremely slow! Please check CUDA installation.")


    # Data Loading and Splitting
    if not os.path.exists(pt_filepath):
        raise FileNotFoundError(f"Cannot find {pt_filepath}! Please run dataset.py first to generate the tensor data.")

    full_dataset = GomokuFastDataset(pt_filepath)
    
    # Split into training and validation sets
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset split complete: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # Initialize Model, Loss Function, and Optimizer

    # Initialize ResNet with 5 residual blocks and 64 feature channels
    model = GomokuResNet(num_blocks=5, num_filters=64).to(device)
    
    # Since the last layer uses F.log_softmax, we must use NLLLoss (Negative Log Likelihood Loss) here
    # This is mathematically equivalent to outputting raw logits and using CrossEntropyLoss
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []

    # Main Training Loop
    print("Starting training")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for boards, labels in pbar:
            boards, labels = boards.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(boards)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * boards.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update progress bar suffix in real-time
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = 100 * correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # No need to calculate gradients during validation, saves VRAM and speeds up
            for boards, labels in val_loader:
                boards, labels = boards.to(device), labels.to(device)
                outputs = model(boards)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * boards.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100 * correct_val / total_val

        history_train_loss.append(epoch_train_loss)
        history_val_loss.append(epoch_val_loss)
        history_train_acc.append(epoch_train_acc)
        history_val_acc.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        # Save Best Model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Found a better model! Saved to '{save_path}' (Accuracy: {best_val_acc:.2f}%)")

    print("Training completed!")

    plot_training_history(history_train_loss, history_val_loss, history_train_acc, history_val_acc)

if __name__ == "__main__":
    train_model()