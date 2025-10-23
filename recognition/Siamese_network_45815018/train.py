import torch
import torch.nn as nn
import torch.functional as F
import dataset
import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model Parameters
batch_size = 128
num_epochs = 20
learning_rate = 1e-3

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
  torch.cuda.manual_seed(42)

train_loader, test_loader = dataset.train_and_validate_loaders('./image', 
                                                               './train-metadata.csv', 
                                                               batch_size)
#model = modules.SiameseNetwork()
#print(model)
#model = model.to(device)
#criterion = modules.ContrastiveLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def create_model(device):
    model = modules.SiameseNetwork()
    return model.to(device)

def create_loss_function():
    return modules.ContrastiveLoss()

def create_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(model, train_loader):
    loss = []
    counter = []
    train_loss = 0.0
    model.train()
    criterion = create_loss_function()
    optimizer = create_optimizer(device, learning_rate)
    for epoch in range(num_epochs):
        for batch_idx, (image1, image2, label) in enumerate(train_loader):
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(image1, image2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            print("Epoch {}\n Current loss {}\n".format(epoch, loss.item()))
            loss.append(loss.item())
            counter.append(epoch)
            train_loss += loss.item()
    plt.plot(counter, loss)
    return model, (train_loss / len(train_loader))

def validate(model, val_loader, threshold=1.0):
    model.eval()
    acc = []
    criterion = create_loss_function()
    all_labels, all_preds, all_distances = [], [], []
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (image1, image2, label) in enumerate(val_loader):
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)
            output1, output2 = model(image1, image2)

            # Distance metric calculation
            euclidean_distance = F.pairwise_distance(output1, output2)
            all_distances.extend(euclidean_distance.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            # Validation loss
            val_loss += criterion(output1, output2, label).item()

            # Get predictions based on threshold
            preds = (euclidean_distance < threshold).float()
            all_preds.extend(preds.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, -torch.tensor(all_distances))  # negative distance = similarity
    except:
        roc_auc = 0.0

    return val_loss / len(val_loader), acc, f1, roc_auc



            


# Can add validation part to test validation error after every epoch as well
# But not added for now
