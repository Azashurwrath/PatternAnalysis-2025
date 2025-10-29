import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
    Class module for the Siamese network
"""
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Dropout2d(p=0.4),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward_one(self, x):
        # Input goes through convolution
        output = self.conv(x)
        # Modify shape to be used in fully connected layer
        output = output.view(output.size(0), -1)
        # Put modified input into fully connected
        output = self.fc(output)
        # Return learn't feature space
        output = F.normalize(output, p=2, dim=1)
        return output

    def forward(self, img1, img2):
        # Get features of first image
        output1 = self.forward_one(img1)
        # Get features of second image
        output2 = self.forward_one(img2)
        # Return both feature vectors
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # cosine distance in [0, 2] for unit vectors
        cos_dist = 1 - F.cosine_similarity(emb1, emb2)   # shape [B]
        label = label.float()
        pos = (1 - label) * (cos_dist ** 2)              # pull same-class together
        neg = label * torch.clamp(self.margin - cos_dist, min=0.0) ** 2
        return (pos + neg).mean()