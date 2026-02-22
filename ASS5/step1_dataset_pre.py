import json
import random
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import glob
import os


# Load dataset
DATA_DIR = "./Potsdam-GeoTif" 
all_files = glob.glob(os.path.join(DATA_DIR, "*.tif"))

# Setting a random seed
random.seed(18)
NUM_SAMPLES = 10000
sampled_files = random.sample(all_files, min(NUM_SAMPLES, len(all_files)))
print(f"Total sampled files: {len(sampled_files)}")

# Splitted to 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(kf.split(sampled_files))
train_idx = np.concatenate([fold_indices[0][1], fold_indices[1][1], fold_indices[2][1]])
val_idx = fold_indices[3][1]
test_idx = fold_indices[4][1]

train_files = [sampled_files[i] for i in train_idx]
val_files = [sampled_files[i] for i in val_idx]
test_files = [sampled_files[i] for i in test_idx]

# Save the splits to a JSON file
splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

with open('data_splits.json', 'w') as f:
    json.dump(splits, f)

print("Data splits saved to data_splits.json successfully.")

# Visualization

# excluding the specific one
excluded_file = "0000000224-0000042784.tif"
valid_files = [f for f in sampled_files if excluded_file not in f]
selected_image_path = random.choice(valid_files)
print(f"Selected image for visualization: {os.path.basename(selected_image_path)}")

with rasterio.open(selected_image_path) as src:
    data = src.read()

# Extract RGB
# Reorder from (C, H, W) to (H, W, C) for matplotlib and normalize for visualization
rgb_image = data[0:3, :, :].transpose(1, 2, 0).astype(np.float32)
rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image) + 1e-8)

# Extract elevation and labels
elevation_band = data[4, :, :]
label_band = data[5, :, :].astype(int)

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

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot RGB Image
axes[0].imshow(rgb_image)
axes[0].set_title('RGB Image')
axes[0].axis('off')

# Plot Elevation
im_elev = axes[1].imshow(elevation_band, cmap='terrain')
axes[1].set_title('Elevation Band')
axes[1].axis('off')
fig.colorbar(im_elev, ax=axes[1], fraction=0.046, pad=0.04, label='Elevation')

# Plot Labels
# vmin=0, vmax=5 ensures colors map strictly to class indices 0-5
im_label = axes[2].imshow(label_band, cmap=cmap_custom, vmin=0, vmax=5)
axes[2].set_title('Target Labels')
axes[2].axis('off')

# Create a colorbar matched to the 6 discrete classes
tick_positions = [0.4, 1.25, 2.1, 2.9, 3.75, 4.6]
cbar = fig.colorbar(im_label, ax=axes[2], ticks=tick_positions, fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(class_names)

plt.tight_layout()
plt.show()