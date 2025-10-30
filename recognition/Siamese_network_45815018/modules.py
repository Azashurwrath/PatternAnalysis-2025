import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
    Class module for the Siamese network
"""
class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
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

        self.embedding = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.classifer = nn.Sequential(
          nn.Dropout(p=0.3),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 1)
      )

    def forward(self, img):
        # Input goes through convolution
        output = self.conv(img)
        # Modify shape to be used in fully connected layer
        output = output.view(output.size(0), -1)
        # Put modified input into fully connected
        embedding = self.embedding(output)

        logits = self.classifer(embedding)

        return logits

class SiameseClassifier(nn.Module):
  def __init__(self):
      super(SiameseClassifier, self).__init__()

      self.features = Embeddings()

  def forward(self, x):

      logits = self.features(x)

      return logits