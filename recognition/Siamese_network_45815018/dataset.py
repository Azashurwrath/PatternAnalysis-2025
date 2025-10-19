# Imports to create dataset
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import os
from glob import glob
import pandas as pd
import random

# Custom dataset class
class SiameseDataset(Dataset):
    def __init__(self, folder_path, csv, transform=None):
        # Initialization
        self.image_paths = sorted(glob(os.path.join(folder_path, "*.png")))
        self.targets = csv[['target']]
        self.transform = transform
        # Create an index dictionary for different classes to use to create image pairs
        self.class_dict = {label: group.index.tolist() 
                           for label, group in csv.groupby('target')}
        self.classes = list(self.class_dict.keys())

    # Get dataset length
    def __len__(self):
        return len(self.image_paths)
    
    # Get an image pair
    def __getitem__(self, idx):
        # Anchor Image and Label
        img1 = Image.open(self.image_paths[idx]).convert("L")
        img_class = self.targets[idx]

        if random() < 0.5:
            # Positive pair
            idx2 = random.choice(self.class_dict[img_class])
            label_pair = 0 # Same class
        else:
            # Negative pair
            neg_class = random.choice([c for c in self.classes if c != img_class])
            idx2 = random.choice(self.class_dict[img_class])
            label_pair = 1 # Different classes
        
        img2 = Image.open(self.image_paths[idx2]).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label_pair
    
"""
Custom function using dataset class and given file path, transformations, 
batch_size and if dataset is shuffled return a data loader to use in torch models
that has even amount of pairs of classes
"""

def get_loader(file_path, csv_path, batch_size, transform):

    temp_csv = pd.read_csv(csv_path)
    temp_csv = temp_csv.sort_values(by='isic_id')
    class_count = temp_csv['target'].value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in class_count.items()}
    sample_weights = [class_weights[targets] for targets in temp_csv['target']]

    # Create dataset using custom class
    dataset = SiameseDataset(file_path, temp_csv, transform=transform)

    # Unbalanced dataset need to create sampler to achieve more balanced sampling
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), replacement=True)

    # Get data loader using given batch_size and shuffle condition
    loader = DataLoader(dataset, batch_size, sampler=sampler)

    # Return loader
    return loader