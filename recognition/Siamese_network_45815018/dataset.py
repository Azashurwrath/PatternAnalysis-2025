# Imports to create dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# For reproducibility
random.seed(42)

# Global Variables
split_ratio = 0.2
batch_size = 128

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize all images to a standard size
    transforms.RandomHorizontalFlip(),        # Random flip for augmentation
    transforms.RandomVerticalFlip(),          # Optional vertical flip
    transforms.RandomRotation(20),            # Random rotations ±20 degrees
    transforms.ToTensor(),                     # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1] for stability
])

# Validation/test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom dataset class
class SiameseDataset(Dataset):
    def __init__(self, img_paths, targets, class_dict, transforms=None):
        # Initialization
        self.image_paths = img_paths
        self.targets = targets
        self.transform = transforms
        # Create an index dictionary for different classes to use to create image pairs
        self.class_dict = class_dict
        self.classes = list(self.class_dict.keys())
        self.pairs = self._create_pairs()

        print(f"Positive pairs: {len([p for p in self.pairs if p[2]==0])}")
        print(f"Negative pairs: {len([p for p in self.pairs if p[2]==1])}")
        print(f"Total pairs: {len(self.pairs)}")

    def _create_pairs(self):
        """Create an equal number of positive and negative pairs."""
        positive_pairs = []
        negative_pairs = []

        # Positive pairs
        for cls, indices in self.class_dict.items():
            if len(indices) < 2:
                continue  # Need at least 2 images to form a positive pair
            # Shuffle and pair sequentially
            shuffled = indices.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled) - 1, 2):
                positive_pairs.append((shuffled[i], shuffled[i+1], 0))  # 0 = same class

        # Negative pairs
        all_indices = list(range(len(self.image_paths)))
        while len(negative_pairs) < len(positive_pairs):
            idx1 = random.choice(all_indices)
            label1 = self.targets[idx1]
            neg_class = random.choice([c for c in self.classes if c != label1])
            idx2 = random.choice(self.class_dict[neg_class])
            negative_pairs.append((idx1, idx2, 1))  # 1 = different class

        # Combine and shuffle
        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)
        return pairs

    # Get dataset length
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]

        img1 = Image.open(self.image_paths[idx1]).convert("L")
        img2 = Image.open(self.image_paths[idx2]).convert("L")

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
    temp_csv = temp_csv[['isic_id', 'target']]
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
    # Combine file paths for ease of use
    csv = combine_file_paths(file_path, csv_path)
    # Get a validation set out of the training set
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        csv['image_path'].to_list(),
        csv['target'].to_list(),
        test_size=split_ratio,
        stratify=csv['target'],
        random_state=42
    )
    # Get dictionary of class indexes for both dataset
    train_dict = class_dict(train_labels)
    val_dict = class_dict(val_labels)

    # Create Dataset
    train_dataset = SiameseDataset(train_paths, train_labels, train_dict, transforms=train_transform)
    val_dataset = SiameseDataset(val_paths, val_labels, val_dict, transforms=val_transform)

    # Weighted sampler based on original image classes
    anchor_classes = [train_labels[idx1] for idx1, idx2, _ in train_dataset.pairs]
    counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    pair_weights = [class_weights[cls] for cls in anchor_classes]

    sampler = WeightedRandomSampler(weights=pair_weights, num_samples=len(pair_weights), replacement=True)

    # Create Data loaders
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Return data loaders
    return train_loader, val_loader