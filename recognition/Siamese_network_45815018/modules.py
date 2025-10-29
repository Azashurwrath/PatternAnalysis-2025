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
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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
        """
        margin:    separation margin between negative pairs
        use_cosine: if True, use cosine distance; else use Euclidean distance
        """
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Loss function
        label = label.float()
        # Euclidean distance
        diff = emb1 - emb2
        dist_sq = torch.sum(diff * diff, dim=1)
        dist = torch.sqrt(dist_sq + 1e-8)

        # --- contrastive loss ---
        pos = label * dist_sq
        # Negative (different) -> push apart until margin: 
        #(1 - label) * max(0,m - d)^2
        neg = (1 - label) * torch.clamp(self.margin - dist, min=0.0).pow(2)
        loss = torch.mean(pos + neg)
        return loss