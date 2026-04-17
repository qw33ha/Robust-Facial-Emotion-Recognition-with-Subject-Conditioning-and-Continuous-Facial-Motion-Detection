"""
training/train_fer2013_shallow_cnn48.py

Purpose:
    Train a shallow CNN (one Conv per stage, 4 stages) on FER2013 for static
    7-class emotion classification. Produces weights/best_fer2013_shallow_cnn48.pt.

Pipeline:
    - Load 48x48 grayscale face images from ./FER2013/train and ./FER2013/test.
    - Split 10% of the training set as validation.
    - Train ShallowCNN48 (4 x Conv+BN+ReLU+MaxPool, then a 3-layer MLP head).
    - 20 epochs, Adam lr=1e-3, CrossEntropy.
    - Keep the checkpoint with the best validation accuracy.
    - After training, evaluate on the test set and save both the confusion
      matrix and the training curve as PNGs.

Input:
    ./FER2013/train/<class>/*.jpg
    ./FER2013/test/<class>/*.jpg

Output:
    best_fer2013_shallow_cnn48.pt   Best checkpoint.
    fer_confusion_matrix.png        Confusion matrix on the test set.
    fer_training_curve.png          Train / val accuracy curve.

Usage (run from project root):
    python training/train_fer2013_shallow_cnn48.py

Note:
    The "ResNet-18" referenced in the milestone report is in fact this
    ShallowCNN48 (a simplified 4-Conv-block CNN, not He et al.'s ResNet-18).
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Subset

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================================================
# 1. Config
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_ROOT = str(PROJECT_ROOT / "FER2013")   # path to the FER2013 dataset root
TRAIN_DIR = f"{DATA_ROOT}/train"
TEST_DIR = f"{DATA_ROOT}/test"

WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = str(WEIGHTS_DIR / "best_fer2013_shallow_cnn48.pt")
CONFUSION_MATRIX_PATH = str(OUTPUTS_DIR / "fer_confusion_matrix.png")
TRAINING_CURVE_PATH = str(OUTPUTS_DIR / "fer_training_curve.png")

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
SEED = 42
NUM_CLASSES = 7
VAL_RATIO = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. Seed
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================================================
# 3. Transforms
# =========================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =========================================================
# 4. Model
# =========================================================

class ShallowCNN48(nn.Module):
    """
    Shallow 4-stage CNN for 48x48 grayscale FER2013 input.

    - Single Conv per stage (Conv + BN + ReLU + MaxPool).
    - Channels: 32 -> 64 -> 128 -> 256.
    - No feature-map dropout.
    - Feature map is flattened directly (no adaptive pool), so the input
      resolution is effectively hard-coded to 48x48.
    - ~1.0M parameters.
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 48x48
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 24x24

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 24x24
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 12x12

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 12x12
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 6x6

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 6x6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 3x3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 5. Data
# =========================================================

def load_data():
    train_dataset_full_aug = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    train_dataset_full_plain = datasets.ImageFolder(TRAIN_DIR, transform=test_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    class_names = train_dataset_full_aug.classes
    print("Classes:", class_names)

    num_train = len(train_dataset_full_aug)
    indices = list(range(num_train))
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    val_size = int(num_train * VAL_RATIO)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(train_dataset_full_aug, train_indices)
    val_dataset = Subset(train_dataset_full_plain, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_names

# =========================================================
# 6. Train / Eval
# =========================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds

# =========================================================
# 7. Plot confusion matrix
# =========================================================

def plot_confusion_matrix(cm, class_names, output_path=CONFUSION_MATRIX_PATH):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="FER Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# 8. Main
# =========================================================

def main():
    print("Using device:", DEVICE)

    train_loader, val_loader, test_loader, class_names = load_data()

    model = ShallowCNN48(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_model_path = BEST_MODEL_PATH

    history = {
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Saved best model to: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_loss, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion, optimizer=None)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, class_names, output_path=CONFUSION_MATRIX_PATH)
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_PATH, dpi=200)
    plt.close()

    print(f"Saved training curve to {TRAINING_CURVE_PATH}")


if __name__ == "__main__":
    main()
