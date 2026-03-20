import os
import random

def extract_and_sample_data(root_dir, output_file, target_samples=500000):
    all_lines = []
    
    print("Starting to traverse directories and extract data")

    # os.walk recursively goes through all sub-directories
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # We only care about the files containing the actual board-to-action data
            if filename.endswith('.txt.board2action.txt'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    all_lines.extend(lines)
                    
    print(f"\nExtraction complete! Found {len(all_lines)} raw samples in total.")
    
    # Remove duplicates.
    unique_lines = list(set(all_lines))
    print(f"Remaining unique samples after deduplication: {len(unique_lines)}")
    
    # Shuffle the data randomly. 
    # This is crucial for deep learning to prevent the model from learning sequential biases.
    print("Shuffling the dataset")
    random.shuffle(unique_lines)
    
    sampled_lines = unique_lines[:target_samples]
    print(f"Writing {len(sampled_lines)} samples to {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(line + '\n' for line in sampled_lines)
        
    print("Done! Dataset is ready.")

if __name__ == "__main__":
    data_folder = './gomoku/divided/'
    final_dataset_name = 'gomoku_sampled_500k.txt'
    
    extract_and_sample_data(data_folder, final_dataset_name, target_samples=500000)