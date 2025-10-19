# Imports to create dataset
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob

# Custom ISIC dataset class
class ISICDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        # Initialization
        self.image_paths = sorted(glob(os.path.join(folder_path, "*.png")))
        self.transform = transform

    # Get dataset length
    def __len__(self):
        return len(self.image_paths)
    
    # Get an image
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img
    
"""
Custom function using dataset class and given file path, transformations, 
batch_size and if dataset is shuffled return a data loader to use in torch models
"""

def get_loader(file_path, transforms, batch_size, shuffle=False):

    # Create dataset using custom class
    dataset = ISICDataset(file_path, transform=transforms)

    # Get data loader using given batch_size and shuffle condition
    loader = DataLoader(dataset, batch_size, shuffle=shuffle)

    # Return loader
    return loader
