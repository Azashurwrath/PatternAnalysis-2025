import torch
from dataset import train_and_validate_loaders
from modules import SiameseClassifier
from params import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_curve, confusion_matrix
from pathlib import Path

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model file path
root_dir = Path(__file__).resolve().parent
load_path = root_dir / "data" / "best_model.pth"

"""
    Function that trains the model and takes 3 inputs:
        model: that has been trained
        test_loader: loader than contains test images and labels
"""
def predict(model, test_loader):
    model.eval()

    # Setup
    all_labels = []
    all_probs = []
    all_preds = []

    # Evaluation model
    with torch.no_grad():
        for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                # Receive probabilities
                logits = model(images)

                # Predictions
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                # Save values in cpu memory
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().detach().numpy())
                all_preds.extend(preds.cpu().numpy())
    
    # Convert to numpy arrays for calculations
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Get AUC Score
    test_auc_score = roc_auc_score(all_labels, all_probs)
    print(f"Test AUC Score={test_auc_score:.4f}")
    print("Test Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='blue', label=f"AUC Score = {test_auc_score:.4f}")
    plt.plot([0,1], [0,1], color='black', linestyle='--', label='Random CLassifier (0.5 AUC)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()

    # Print accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"Overall Accuracy for Test set: {accuracy:.4f}")

    # Print confusion matrix for class imbalance display
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

# Get test data
_, _, test_loader = train_and_validate_loaders()

# Load in save model
model = SiameseClassifier()
model.load_state_dict(torch.load(load_path, map_location="cpu"))
model = model.to(device)

# Run prediction on test data
predict(model, test_loader)