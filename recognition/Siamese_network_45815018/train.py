import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import train_and_validate_loaders
from modules import SiameseNetwork, ContrastiveLoss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model Parameters
batch_size = 512
num_epochs = 10
learning_rate = 1e-4
margin = 1

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
  torch.cuda.manual_seed(42)

train_loader, val_loader = train_and_validate_loaders()

def create_model(device):
    model = SiameseNetwork()
    return model.to(device)

def create_loss_function(margin):
    return ContrastiveLoss(margin=margin)

def create_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

def train(model, train_loader, val_loader):
    # --- setup ---
    criterion = ContrastiveLoss(margin=margin)  # start with Euclidean
    optimizer = create_optimizer(model, learning_rate)          # e.g. 1e-4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)

    loss_list, val_loss_list, counter = [], [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Refresh negative pairs every 5 epochs
        if epoch > 0:
            train_loader.dataset.refresh_pairs(epoch)

        model.train()
        total_loss, all_dists, all_labels = 0.0, [], []

        for image1, image2, label, _ in train_loader:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)
            optimizer.zero_grad()

            output1, output2 = model(image1, image2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                d = torch.norm(output1 - output2, p=2, dim=1)
                all_dists.extend(d.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        # --- metrics ---
        avg_train_loss = total_loss / len(train_loader)
        loss_list.append(avg_train_loss)
        counter.append(epoch + 1)

        all_dists, all_labels = np.array(all_dists), np.array(all_labels)
        pos_mask = all_labels == 1 # same class
        neg_mask = all_labels == 0 # diff class
        pos_mean_d = np.mean(all_dists[pos_mask]) if pos_mask.any() else float('nan')
        neg_mean_d = np.mean(all_dists[neg_mask]) if neg_mask.any() else float('nan')
        neg_inside = np.mean(all_dists[neg_mask] < margin) if neg_mask.any() else float('nan')

        print(f"\nEpoch {epoch}")
        print(f"Train Loss={avg_train_loss:.4f}, "
              f"pos_mean_d={pos_mean_d:.3f}, neg_mean_d={neg_mean_d:.3f}, "
              f"%neg_inside_margin={100*neg_inside:.1f}%")

        # --- validation ---
        model.eval()
        val_loss, val_dists, val_labels = 0.0, [], []
        with torch.no_grad():
            for image1, image2, label, _ in val_loader:
                image1, image2, label = image1.to(device), image2.to(device), label.to(device)
                output1, output2 = model(image1, image2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()

                d = torch.norm(output1 - output2, p=2, dim=1)
                val_dists.extend(d.cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        val_dists, val_labels = np.array(val_dists), np.array(val_labels)
        pos_mask = val_labels == 1
        neg_mask = val_labels == 0
        pos_mean_d = np.mean(val_dists[pos_mask]) if pos_mask.any() else float('nan')
        neg_mean_d = np.mean(val_dists[neg_mask]) if neg_mask.any() else float('nan')

        print(f"Val Loss={avg_val_loss:.4f}, "
              f"Val pos_mean_d={pos_mean_d:.3f}, Val neg_mean_d={neg_mean_d:.3f}")

        # --- checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # --- plot ---
    plt.plot(counter, loss_list, label="Train Loss")
    plt.plot(counter, val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    return model

def validate(model, val_loader):
    model.eval()
    all_labels, all_distances, all_pairs = [], [], []
    with torch.no_grad():
        for image1, image2, label, (idx1, idx2) in val_loader:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)
            output1, output2 = model(image1, image2)
            # Distance metric calculation
            distances = torch.norm(output1 - output2, p=2, dim=1)
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_pairs.extend(zip(idx1, idx2))
    return all_distances, all_labels, all_pairs


def threshold_sweep(distances, labels, num_thresholds=200):
    distances = np.array(distances)
    labels = np.array(labels)

    auc = roc_auc_score(labels, -distances)
    print(f"ROC AUC: {auc:.3f}")

    thresholds = np.linspace(distances.min(), distances.max(), num_thresholds)

    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []

    for t in thresholds:
        pred = (distances < t).astype(int)
        f1_scores.append(f1_score(labels, pred))
        accuracies.append(accuracy_score(labels, pred))
        precisions.append(precision_score(labels, pred, zero_division=0))
        recalls.append(recall_score(labels, pred, zero_division=0))

    f1_scores = np.array(f1_scores)
    accuracies = np.array(accuracies)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    metrics_dict = {
        'threshold': best_threshold,
        'F1': f1_scores[best_idx],
        'accuracy': accuracies[best_idx],
        'precision': precisions[best_idx],
        'recall': recalls[best_idx]
    }
    print("Best threshold:", best_threshold)
    print("Metrics at best threshold:", metrics_dict)

    fpr, tpr, _ = roc_curve(labels, -distances)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()

    precision, recall, _ = precision_recall_curve(labels, -distances)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label ='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

    return best_threshold, metrics_dict, thresholds, f1_score

base = create_model(device)
trained_model = train(base, train_loader, val_loader)
distances, labels, pairs = validate(trained_model, val_loader)

same = [d for d, l in zip(distances, labels) if l == 0]
diff = [d for d, l in zip(distances, labels) if l == 1]

print("mean same:", np.mean([d for d,l in zip(distances, labels) if l==0]))
print("mean diff:", np.mean([d for d,l in zip(distances, labels) if l==1]))

plt.hist(same, bins=50, alpha=0.5, label='Same class')
plt.hist(diff, bins=50, alpha=0.5, label='Different class')
plt.legend()
plt.xlabel("Cosine Distance")
plt.ylabel("Count")
plt.title("Validation Distance Distribution")
plt.show()

best_t, metrics, thresholds, f1_scores = threshold_sweep(distances, labels, num_thresholds=200)