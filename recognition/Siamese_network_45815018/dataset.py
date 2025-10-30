# Imports to create dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from params import *

# For reproducibility
random.seed(42)
gen = torch.Generator()
gen.manual_seed(42)
random_state = 42

# Global Variables
split_val = 0.15
split_test = 0.10

file_path = '~/data/image'
csv_path = '~/data/train-metadata.csv'

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize all images to a standard size
    transforms.RandomHorizontalFlip(p=0.5),        # Random flip for augmentation
    transforms.RandomVerticalFlip(p=0.5),          # Optional vertical flip
    transforms.RandomRotation(30),            # Random rotations ±30 degrees
    transforms.ToTensor(),                     # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5])  # Normalize for stability
])

# Validation/test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5])
])

"""
    Custom Dataset for an ISIC 2020 dataset
"""
class SiameseDataset(Dataset):
    def __init__(self, img_paths, targets, transforms=None):
        # Variables for dataset
        self.image_paths = img_paths
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def _load_img(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        img = self._load_img(idx)
        label = self.targets[idx]

        if self.transforms:
            img = self.transforms(img)

        return img, torch.tensor(label, dtype=torch.float32)

"""
    Combine the metadata csv and file paths to images into one file and row for easy access and use
"""
def combine_file_paths(image_path, csv_path):
    # Read csv
    temp_csv = pd.read_csv(csv_path)
    # Isolate only image ids and labels
    temp_csv = temp_csv[['isic_id', 'target', 'patient_id']]
    # Sort rows by image ids
    temp_csv = temp_csv.sort_values(by='isic_id')
    # Append image path to given image id for each row
    temp_csv['image_path'] = temp_csv['isic_id'].apply(lambda x: os.path.join(image_path, f"{x}.jpg"))
    return temp_csv

"""
    Separates a target list into a dictionary of classes containing indexes of each class value
"""
def class_dict(targets):
    classes = {}
    # Iterate through each value and append to the given class in the dictionary
    for idx, lbl in enumerate(targets):
        classes.setdefault(lbl, []).append(idx)
    return classes

"""
    Custom function that allows a training dataset
    to be split into train and validate data loaders
"""
def train_and_validate_loaders():
    # Combine file paths with metadata
    csv = combine_file_paths(file_path, csv_path)

    # Split into three seperate csv files to use for train:val:test
    train_csv, temp_csv = train_test_split(
        csv,
        test_size=(1 - (split_test + split_val)),
        stratify=csv['target'],
        random_state=random_state
    )
    val_ratio_adjusted = split_val / (1 - split_test)
    val_csv, test_csv = train_test_split(
        temp_csv,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_csv['target'],
        random_state=random_state
    )

    # Prepare lists and class dictionaries
    train_paths = train_csv["image_path"].to_list()
    train_labels = train_csv["target"].to_list()

    val_paths = val_csv["image_path"].to_list()
    val_labels = val_csv["target"].to_list()

    test_paths = val_csv["image_path"].to_list()
    test_labels = val_csv["target"].to_list()

    train_dict = class_dict(train_labels)

    # Create Datasets
    train_dataset = SiameseDataset(train_paths, train_labels,
                                   transforms=train_transform)

    val_dataset = SiameseDataset(val_paths, val_labels,
                                 transforms=val_transform)

    test_dataset = SiameseDataset(test_paths, test_labels,
                                  transforms=val_transform)

    # Create sampler for training dataset
    class_count = {key: len(count) for key, count in train_dict.items()}
    class_weights = {key: 1.0 / count for key, count in class_count.items()}
    sample_weights = [class_weights[targets] for targets in train_labels]

    # Unbalanced dataset need to create sampler to achieve more balanced sampling
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

