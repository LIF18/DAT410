import torch
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def preprocess_and_save_dataset(txt_filepath, pt_filepath):
    # we only need to run this function once.

    print(f"Starting to read {txt_filepath} and converting to tensors (this may takes a few minutes)")
    
    with open(txt_filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    num_samples = len(lines)
    # Pre-allocating memory space.
    all_boards = torch.zeros((num_samples, 3, 15, 15), dtype=torch.float32)
    all_labels = torch.zeros((num_samples,), dtype=torch.long)
    
    for i, line in tqdm(enumerate(lines), total=num_samples, desc="Processing"):
        board_str, move_str = line.split(':')
        
        # Process input X
        row_ints = [int(val) for val in board_str.split(',')]
        for y in range(15):
            row_val = row_ints[y]
            for x in range(15):
                bits = (row_val >> (2 * x)) & 0b11
                if bits == 1:
                    all_boards[i, 0, y, x] = 1.0  # black
                elif bits == 2:
                    all_boards[i, 1, y, x] = 1.0  # white
                else:
                    all_boards[i, 2, y, x] = 1.0  # empty
                    
        # Process lables Y
        move_x = ord(move_str[0]) - ord('a')
        move_y = int(move_str[1:]) - 1
        all_labels[i] = move_y * 15 + move_x
        
    print(f"Conversion complete! Saving to {pt_filepath}")
    torch.save({'boards': all_boards, 'labels': all_labels}, pt_filepath)
    print("Saved successfully! This .pt file will be loaded directly for future training")


class GomokuFastDataset(Dataset):

    # High-speed Dataset for training. Directly loads pre-processed .pt files.
    def __init__(self, pt_filepath):
        print(f"Loading dataset into memory: {pt_filepath}")
        data = torch.load(pt_filepath)
        self.boards = data['boards']
        self.labels = data['labels']
        print(f"Loading complete! Total of {len(self.labels)} data entries found.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.boards[idx], self.labels[idx]

if __name__ == "__main__":
    txt_file = "gomoku_sampled_500k.txt"
    pt_file = "gomoku_500k.pt"
    
    # we only need to run this function once.
    preprocess_and_save_dataset(txt_file, pt_file)
    