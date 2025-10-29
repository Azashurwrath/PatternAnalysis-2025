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
class SiameseDataset(Dataset):
    def __init__(self, img_paths, targets, patient_ids, class_dict, transforms=None):
        # Initialization
        self.image_paths = img_paths
        self.targets = targets
        self.patient_ids = patient_ids
        self.transform = transforms
        # Create an index dictionary for different classes to use to create image pairs
        self.class_dict = class_dict
        self.classes = list(self.class_dict.keys())
        self.pairs = self._create_pairs()

    """
        Create balanced positive and negative pairs.
        Ensures:
          • Positive pairs = same class, different patients
          • Negative pairs = different class, different patients
          • Equal benign and malignant anchors (oversampling malignant)
          • Keeps large pair count (~full dataset * ratio (for training speed))
        """
    def _create_pairs(self):
        
        positive_pairs, negative_pairs = [], []

        # --- Separate class indices ---
        benign_idxs = [i for i, y in enumerate(self.targets) if y == 0]
        malig_idxs  = [i for i, y in enumerate(self.targets) if y == 1]
        class_groups = {0: benign_idxs, 1: malig_idxs}

        # --- Oversample minority class so both anchor sets are same length ---
        sample_n = len(benign_idxs)
        benign_sample = random.sample(benign_idxs, sample_n)
        malig_sample = (
            random.choices(malig_idxs, k=sample_n)
            if len(malig_idxs) < sample_n
            else random.sample(malig_idxs, sample_n)
        )

        # ==============================================================
        # --- POSITIVE PAIRS: same class, different patients ---
        # ==============================================================
        for cls, indices in class_groups.items():
            # Pick anchors: oversampled malignant, full benign
            anchors = benign_sample if cls == 0 else malig_sample
            shuffled = indices.copy()
            random.shuffle(shuffled)

            for idx1 in anchors:
                pid1 = self.patient_ids[idx1]
                # Same class, different patient
                candidates = [idx2 for idx2 in shuffled if self.patient_ids[idx2] != pid1]
                if not candidates:
                    continue
                idx2 = random.choice(candidates)
                positive_pairs.append((idx1, idx2, 0))  # 0 = same class (diff patient)

        # ==============================================================
        # --- NEGATIVE PAIRS: different class, different patients ---
        # ==============================================================
        anchor_pool = benign_sample + malig_sample
        random.shuffle(anchor_pool)

        while len(negative_pairs) < len(positive_pairs):
            idx1 = random.choice(anchor_pool)
            label1 = self.targets[idx1]
            pid1 = self.patient_ids[idx1]

            neg_class = 1 - label1
            idx2 = random.choice(self.class_dict[neg_class])
            pid2 = self.patient_ids[idx2]
            if pid1 == pid2:
                continue

            negative_pairs.append((idx1, idx2, 1))  # 1 = different class

        # --- Combine and shuffle ---
        pair_ratio = 0.25  # keep 25% of all pairs

        # compute how many positives/negatives to keep
        keep_pos = int(len(positive_pairs) * pair_ratio)
        keep_neg = int(len(negative_pairs) * pair_ratio)

        positive_pairs = random.sample(positive_pairs, keep_pos)
        negative_pairs = random.sample(negative_pairs, keep_neg)

        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)

        return pairs

    # Get dataset length
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]

        img1 = Image.open(self.image_paths[idx1]).convert("RGB")
        img2 = Image.open(self.image_paths[idx2]).convert("RGB")

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
        train_dict, transforms=train_transform
    )
    val_dataset = SiameseDataset(
        val_paths, val_labels, val_pids,
        val_dict, transforms=val_transform
    )

    # Weighted sampler for class balancing
    anchor_classes = [train_labels[idx1] for idx1, _, _ in train_dataset.pairs]
    counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    pair_weights = [class_weights[cls] for cls in anchor_classes]

    sampler = WeightedRandomSampler(
        weights=pair_weights,
        num_samples=len(pair_weights),
        replacement=True,
        generator=gen
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    return train_loader, val_loader
