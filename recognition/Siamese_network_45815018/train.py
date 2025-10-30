import torch
import torch.nn as nn
from dataset import train_and_validate_loaders
from modules import SiameseClassifier
from params import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model file path
root_dir = Path(__file__).resolve().parent
load_path = root_dir / "data" / "best_model.pth"

"""
    Function to create the model and send it to the connect device with 1 input:
        device: the device the model with be sent to for faster processing - GPU or CPU
"""
def create_model(device):
    model = SiameseClassifier()
    return model.to(device)

"""
    Function to create the optimizer with 2 inputs:
        model: model that is to be trained
        learning_rate: learning rate of the model
"""
def create_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

"""
    Function that trains the model and takes 3 inputs:
        model: model that is to be trained
        train_loader: loader than contains training images and labels
        val_loader: loader than contains validation images and labels
"""
def train(model, train_loader, val_loader):
    # --- setup ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = create_optimizer(model, learning_rate)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    counter = []
    train_auc = []
    val_auc = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        # Epoch setip
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        train_labels = []
        train_probs = []

        for images, labels in train_loader:
            # Training
            images, labels = images.to(device), labels.to(device).float()

            # Getting training loss
            optimizer.zero_grad()
            logits = model(images)
            logits = logits.squeeze()
            loss = criterion(logits, labels)

            # Optimizing
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            # Save values for plotting
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(probs.detach().cpu().numpy())

        # --- metrics ---
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        counter.append(epoch+1)

        # Printing metrics out
        print(f"\nEpoch {epoch}")
        print(f"Train Loss={avg_train_loss:.4f}")
        train_auc_score = roc_auc_score(train_labels, train_probs)
        train_auc.append(train_auc_score)
        print(f"Train AUC Score={train_auc_score:.4f}")
        

        # --- validation ---
        # validation setup
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_labels = []
        val_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()

                # Get validation loss
                logits = model(images)
                logits = logits.squeeze()
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Predictions
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                # Save values for ploting
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.detach().cpu().numpy())

        # Calculate epoch losses
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Print metrics
        print(f"Val Loss={avg_val_loss:.4f}")
        val_preds = (np.array(val_probs) >= 0.5).astype(int)
        val_auc_score = roc_auc_score(val_labels, val_probs)
        val_auc.append(val_auc_score)
        print(f"Val AUC Score={val_auc_score:.4f}")
        # Generate classification report
        print("Validation Classification Report:")
        print(classification_report(val_labels, val_preds, zero_division=0))

    # Save model 
    torch.save(model.state_dict(), load_path)

    # --- plots ---
    # Loss plots
    plt.figure(figsize=(6,5))
    plt.plot(counter, train_losses, label="Train Loss")
    plt.plot(counter, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy plot
    plt.figure(figsize=(6,5))
    plt.plot(counter, train_accuracies, label="Train Accuracy")
    plt.plot(counter, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # AUC plot
    plt.figure(figsize=(6,5))
    plt.plot(counter, train_auc, label="Train AUC Score")
    plt.plot(counter, val_auc, label="Validation AUC Score")
    plt.xlabel("Epoch")
    plt.ylabel("AUC Score")
    plt.title("Training vs Validation AUC Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# Get train and validation loader
train_loader, val_loader, _ = train_and_validate_loaders()

# Get the base untrained model
base = create_model(device)
# Train the model and then save it
trained_model = train(base, train_loader, val_loader)