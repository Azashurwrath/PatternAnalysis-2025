import torch
import torch.nn as nn
import numpy as np


"""
    Class module for the Siamese network
"""
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fcc = nn.Sequential(
            nn.Linear(100352, 4096),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Linear(4096, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
        )

        def forward_one(self, x):
            # Input goes through convolution
            output = self.conv(x)
            # Modify shape to be used in fully connected layer
            output = output.view(output.size()[0], -1)
            # Put modified input into fully connected
            output = self.fcc(output)
            # Return learn't feature space
            return output
        
        def forward(self, img1, img2):
            # Get features of first image
            output1 = self.forward_once(img1)
            # Get features of second image
            output2 = self.forward_once(img2)
            # Return both feature vectors
            return output1, output2
        
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, x, y, z):
        # Using euclidian distance
        diff = x - y
        sq_dist = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(sq_dist)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * sq_dist + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x.size()[0]

        return loss