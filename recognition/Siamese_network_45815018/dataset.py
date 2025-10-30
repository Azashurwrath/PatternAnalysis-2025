# Imports to create dataset
from torch.utils.data import Dataset, DataLoader
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
split_val = 0.15
split_test = 0.15
batch_size = 512
neg_ratio = 1
pairs_per_class = 8000

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

class SiameseDataset(Dataset):
    """
    Patient-leakage-safe Siamese dataset for ISIC 2020.
    --------------------------------------------------
    Precomputes balanced positive & negative pairs and
    refreshes negative pairs every N epochs to maintain learning diversity.

    Args:
        img_paths (list): List of full image paths.
        targets (list[int]): Binary class labels (0=benign, 1=malignant).
        patient_ids (list): Corresponding patient IDs.
        class_dict (dict): Mapping of class -> list of sample indices.
        transform: Torchvision transform.
        pairs_per_class (int): Number of positive pairs per class.
        neg_pos_ratio (float): Negatives per positive (1.0 = equal).
        refresh_every (int): Epoch interval to refresh negatives.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, img_paths, targets, patient_ids, class_dict,
                 transforms=None, pairs_per_class=8000, neg_pos_ratio=1.0,
                 refresh_every=5, seed=42,):
        # Variables for dataset
        self.image_paths = img_paths
        self.targets = targets
        self.patient_ids = patient_ids
        self.class_dict = class_dict
        self.classes = list(class_dict.keys())
        self.transforms = transforms
        self.pairs_per_class = pairs_per_class
        self.neg_pos_ratio = neg_pos_ratio
        self.refresh_every = refresh_every
        self.epoch_count = 0
        self.rng = random.Random(seed)

        # Build initial pairs
        self.pos_pairs = self._build_positive_pairs()
        self.neg_pairs = self._build_negative_pairs()
        self._combine_and_shuffle_pairs()

    # ------------------------------------------------------
    # BUILD FUNCTIONS
    # ------------------------------------------------------
    def _build_positive_pairs(self):
        """Create same-class pairs from different patients."""
        pos_pairs = []
        for c in self.classes:
            indices = self.class_dict[c]
            for _ in range(self.pairs_per_class):
                while True:
                    i, j = self.rng.sample(indices, 2)
                    if self.patient_ids[i] != self.patient_ids[j]:
                        pos_pairs.append((i, j, 1.0))
                        break
        return pos_pairs

    def _build_negative_pairs(self):
        """Create cross-class pairs from different patients."""
        neg_pairs = []
        total_neg = int(len(self.pos_pairs) * self.neg_pos_ratio)
        benign = self.class_dict[0]
        malig = self.class_dict[1]

        for _ in range(total_neg):
            while True:
                i = self.rng.choice(benign)
                j = self.rng.choice(malig)
                if self.patient_ids[i] != self.patient_ids[j]:
                    neg_pairs.append((i, j, 0.0))
                    break
        return neg_pairs

    def _combine_and_shuffle_pairs(self):
        """Merge and shuffle pairs into one list."""
        self.pairs = self.pos_pairs + self.neg_pairs
        self.rng.shuffle(self.pairs)

    # ------------------------------------------------------
    # REFRESH MECHANISM
    # ------------------------------------------------------
    def refresh_pairs(self, epoch=None):
        """
        Refresh negative pairs every `refresh_every` epochs
        to expose the model to new contrastive samples.
        """
        if epoch is not None:
            self.epoch_count = epoch

        if self.epoch_count % self.refresh_every == 0:
            print(f"[Dataset] Refreshing negative pairs at epoch {self.epoch_count}...")
            self.neg_pairs = self._build_negative_pairs()
            self._combine_and_shuffle_pairs()

    # ------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def _load_img(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        img1, img2 = self._load_img(idx1), self._load_img(idx2)

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32), (idx1, idx2)

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
    patient_targets = (
        csv.groupby("patient_id")["target"]
           .agg(lambda x: x.mode()[0])  # use the majority label per patient
           .reset_index()
    )

    train_val_pids, val_pids = train_test_split(
        patient_targets["patient_id"],
        test_size=split_val,
        stratify=patient_targets["target"],
        random_state=42
    )

    train_pids, test_pids = train_test_split(
        train_val_pids,
        test_size=split_val / (1 - split_test),  # adjust fraction relative to remaining data
        stratify=patient_targets.loc[
            patient_targets["patient_id"].isin(train_val_pids), "target"
        ],
        random_state=42
    )

    # Create train/validation splits by patient membership
    train_csv = csv[csv["patient_id"].isin(train_pids)]
    val_csv   = csv[csv["patient_id"].isin(val_pids)]
    test_csv  = csv[csv["patient_id"].isin(test_pids)]

    # Prepare lists and class dictionaries
    train_paths = train_csv["image_path"].to_list()
    train_labels = train_csv["target"].to_list()
    train_pids = train_csv["patient_id"].to_list()

    val_paths = val_csv["image_path"].to_list()
    val_labels = val_csv["target"].to_list()
    val_pids = val_csv["patient_id"].to_list()

    test_paths = val_csv["image_path"].to_list()
    test_labels = val_csv["target"].to_list()
    test_pids = val_csv["patient_id"].to_list()

    train_dict = class_dict(train_labels)
    val_dict   = class_dict(val_labels)
    test_dict  = class_dict(test_labels)

    # Create Datasets
    train_dataset = SiameseDataset(
        train_paths, train_labels, train_pids,
        train_dict, transforms=train_transform,
        pairs_per_class=pairs_per_class, neg_pos_ratio=neg_ratio)
    
    val_dataset = SiameseDataset(
        val_paths, val_labels, val_pids,
        val_dict, transforms=val_transform,
        pairs_per_class=pairs_per_class, neg_pos_ratio=neg_ratio)
    
    test_dataset = SiameseDataset(
        test_paths, test_labels, test_pids,
        test_dict, transforms=val_transform,
        pairs_per_class=pairs_per_class, neg_pos_ratio=neg_ratio)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
