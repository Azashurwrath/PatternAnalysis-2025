# Imports to create dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import torch

# For reproducibility
random.seed(42)
gen = torch.Generator()
gen.manual_seed(42)

# Global Variables
split_ratio = 0.2
batch_size = 512

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize all images to a standard size
    transforms.RandomHorizontalFlip(),        # Random flip for augmentation
    transforms.RandomVerticalFlip(),          # Optional vertical flip
    transforms.RandomRotation(20),            # Random rotations ±20 degrees
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

# Custom dataset class
import random
from torch.utils.data import Dataset
from PIL import Image

class SiameseDataset(Dataset):
    def __init__(self, img_paths, targets, patient_ids, class_dict, transforms=None):
        self.image_paths = img_paths
        self.targets = targets
        self.patient_ids = patient_ids
        self.class_dict = class_dict
        self.classes = list(class_dict.keys())
        self.transform = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx1):
        img1_path = self.image_paths[idx1]
        label1 = self.targets[idx1]
        pid1 = self.patient_ids[idx1]

        # --- Decide positive or negative pair ---
        is_positive = random.random() < 0.5

        if is_positive:
            # Pick from same class, different patient
            candidates = [
                i for i in self.class_dict[label1]
                if self.patient_ids[i] != pid1
            ]
            label = 0
        else:
            # Pick from opposite class, different patient
            neg_class = 1 - label1
            candidates = [
                i for i in self.class_dict[neg_class]
                if self.patient_ids[i] != pid1
            ]
            label = 1

        # Safety check
        if not candidates:
            # Fallback: random other index
            idx2 = random.choice(range(len(self.image_paths)))
        else:
            idx2 = random.choice(candidates)

        img2_path = self.image_paths[idx2]

        # --- Load images ---
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label, (idx1, idx2)

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
    file_path = './image'
    csv_path = './train-metadata.csv'

    # Combine file paths with metadata
    csv = combine_file_paths(file_path, csv_path)

    # Split by patient_id, not by image, to avoid leakage
    # Keep class ratio (≈5:1) using patient-level stratification
    patient_targets = (
        csv.groupby("patient_id")["target"]
           .agg(lambda x: x.mode()[0])  # use the majority label per patient
           .reset_index()
    )

    train_pids, val_pids = train_test_split(
        patient_targets["patient_id"],
        test_size=split_ratio,
        stratify=patient_targets["target"],
        random_state=42
    )

    # Create train/validation splits by patient membership
    train_csv = csv[csv["patient_id"].isin(train_pids)]
    val_csv   = csv[csv["patient_id"].isin(val_pids)]

    # Prepare lists and class dictionaries
    train_paths = train_csv["image_path"].to_list()
    train_labels = train_csv["target"].to_list()
    train_pids = train_csv["patient_id"].to_list()

    val_paths = val_csv["image_path"].to_list()
    val_labels = val_csv["target"].to_list()
    val_pids = val_csv["patient_id"].to_list()

    train_dict = class_dict(train_labels)
    val_dict   = class_dict(val_labels)

    # Create Datasets
    train_dataset = SiameseDataset(
        train_paths, train_labels, train_pids,
        train_dict, transforms=train_transform)
    val_dataset = SiameseDataset(
        val_paths, val_labels, val_pids,
        val_dict, transforms=val_transform)

    # Weighted sampler for anchors (balancing malignant class)
    counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
        generator=gen
    )
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, val_loader
