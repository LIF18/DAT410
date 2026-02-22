import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset

class PotsdamDataset(Dataset):
    def __init__(self, file_paths, input_bands='all'):
        # file_paths: List of file paths
        # input_bands: 'all' (bands 0-4) for Step 3 and 'rgb_ir' (bands 0-3) for Step 2

        self.file_paths = file_paths
        self.input_bands = input_bands

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        
        # Read tif file
        with rasterio.open(img_path) as src:
            data = src.read() 

        # Channels 0, 1, 2, 3, 4, 5 are Red, Green, Blue, IR, Elevation. 
        # The 6th channel (index 5) contains the labels.    
        if self.input_bands == 'rgb_ir':
            # Use Red, Green, Blue, IR (indices 0 to 3) -> shape (4, 224, 224)
            x = data[0:4, :, :].astype(np.float32)
        else:
            # Use all 5 input features (indices 0 to 4) -> shape (5, 224, 224)
            x = data[0:5, :, :].astype(np.float32)
            
        # Target band is index 5
        y = data[5, :, :].astype(np.int64) 
        
        return torch.from_numpy(x), torch.from_numpy(y)

# def augment_batch(images, masks):
#     aug_images, aug_masks = [], []
#     for img, mask in zip(images, masks):
#         # Random Horizontal Flip
#         if random.random() > 0.5:
#             img = TF.hflip(img)
#             mask = TF.hflip(mask)
#         # Random Vertical Flip
#         if random.random() > 0.5:
#             img = TF.vflip(img)
#             mask = TF.vflip(mask)
#         # Random Rotation (0, 90, 180, 270 degrees)
#         k = random.randint(0, 3)
#         if k > 0:
#             img = torch.rot90(img, k, [1, 2])
#             mask = torch.rot90(mask, k, [0, 1])
            
#         aug_images.append(img)
#         aug_masks.append(mask)
#     return torch.stack(aug_images), torch.stack(aug_masks)