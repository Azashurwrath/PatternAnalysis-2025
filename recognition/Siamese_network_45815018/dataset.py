# Imports to create dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# For reproducibility
random.seed(42)

# Global Variables
split_ratio = 0.2

# Transformations
train_transform = transforms.Compose([transforms.ToTensor()])
val_transform = transforms.Compose([transforms.ToTensor()])

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

    # Get dataset length
    def __len__(self):
        return len(self.image_paths)
    
    # Get an image pair
    def __getitem__(self, idx):
        # Anchor Image and Label
        img1 = Image.open(self.image_paths[idx]).convert("L")
        img_class = self.targets[idx]

        # Average split of pairs
        if random.random() < 0.5:
            # Positive pair
            idx2 = random.choice(self.class_dict[img_class])
            label_pair = 0 # Same class
        else:
            # Negative pair
            neg_class = random.choice([c for c in self.classes if c != img_class])
            idx2 = random.choice(self.class_dict[neg_class])
            label_pair = 1 # Different classes
        # Open paired image
        img2 = Image.open(self.image_paths[idx2]).convert("L")
        # Transform images if transformations are applied
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # Return anchor image, a pair image and a label depicting if they are the same class
        return img1, img2, label_pair

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
def train_and_validate_loaders(file_path, csv_path, batch_size):
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

    # Create sampler for training dataset
    class_count = {key: len(count) for key, count in train_dict.items()}
    class_weights = {key: 1.0 / count for key, count in class_count.items()}
    sample_weights = [class_weights[targets] for targets in train_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), replacement=True)
    
    # Create Dataset
    train_dataset = SiameseDataset(train_paths, train_labels, train_dict, transforms=train_transform)
    val_dataset = SiameseDataset(val_paths, val_labels, val_dict, transforms=val_transform)

    # Create Data loaders
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Return data loaders
    return train_loader, val_loader