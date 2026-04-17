"""
training/train_ravdess_cnn_actor_split.py

Purpose:
    Train a deeper VGG-style CNN on RAVDESS grayscale frames using an
    actor-independent (subject-independent) train/val split. Produces
    best_ravdess_actor_split_model.pth.

Pipeline:
    - Load the grayscale frames written to ./output_gray_frames/ by
      preprocess/extract_ravdess_gray_frames.py.
    - Parse actor_id from the filename.
    - Use actors 21/22/23/24 as the validation set; all other actors go
      into the training set.
    - Train VGGStyleCNN128 (4 double-conv stages + adaptive avg pool +
      4-layer MLP, 128x128 grayscale input).
    - 20 epochs, Adam lr=1e-3 + weight decay 1e-4, ReduceLROnPlateau.
    - Every epoch prints val acc and UAR (computed by hand from the
      confusion matrix).
    - Save the checkpoint with the best validation accuracy.

Input:
    ./output_gray_frames/<emotion>/*.png

Output:
    best_ravdess_actor_split_model.pth     Best checkpoint.

Usage (run from project root):
    python training/train_ravdess_cnn_actor_split.py
"""

import re
from pathlib import Path
from collections import Counter

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class VGGStyleCNN128(nn.Module):
    """
    Deeper VGG-style CNN designed for 128x128 (or larger) grayscale input.

    - 4 stages, each with a double Conv (Conv -> ReLU -> [BN] -> Conv ->
      ReLU -> BN -> MaxPool -> Dropout2d(0.25)).
    - Channels: 64 -> 128 -> 256 -> 512.
    - AdaptiveAvgPool2d((2, 2)) before the classifier makes the model
      insensitive to exact input resolution.
    - 4-layer MLP head.
    - ~5.9M parameters.
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def parse_actor_id(filename: str) -> int:
    stem = Path(filename).stem

    m = re.search(r"_actor_(\d+)_", stem)
    if m:
        return int(m.group(1))

    base = stem.split("_frame_")[0]
    parts = base.split("-")
    if len(parts) >= 7:
        return int(parts[6])

    raise ValueError(f"Cannot parse actor id from filename: {filename}")


class RAVDESSFrameDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_name = self.samples[idx]
        image = Image.open(img_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[label_name]
        return image, label


def collect_samples_by_actor(data_dir: str):
    data_path = Path(data_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    class_names = sorted([p.name for p in data_path.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_samples = []
    actor_counter = Counter()

    for class_name in class_names:
        class_dir = data_path / class_name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in image_exts:
                actor_id = parse_actor_id(img_path.name)
                all_samples.append((img_path, class_name, actor_id))
                actor_counter[actor_id] += 1

    return all_samples, class_names, class_to_idx, actor_counter


def split_by_actor(all_samples, val_actor_ids):
    train_samples = []
    val_samples = []

    for img_path, class_name, actor_id in all_samples:
        item = (img_path, class_name)
        if actor_id in val_actor_ids:
            val_samples.append(item)
        else:
            train_samples.append(item)

    return train_samples, val_samples


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for t, p in zip(labels.cpu(), preds.cpu()):
            confusion[t.long(), p.long()] += 1

    val_loss = total_loss / total_samples
    val_acc = total_correct / total_samples

    confusion = confusion.numpy()
    val_uar, per_class_recalls = compute_uar_from_confusion(confusion, num_classes)

    return val_loss, val_acc, val_uar, per_class_recalls

def compute_uar_from_confusion(confusion, num_classes):
    recalls = []

    for i in range(num_classes):
        tp = confusion[i, i]
        fn = confusion[i].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    uar = sum(recalls) / num_classes if num_classes > 0 else 0.0
    return uar, recalls


def main():
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = str(PROJECT_ROOT / "output_gray_frames")
    best_model_path = str(WEIGHTS_DIR / "best_ravdess_actor_split_model.pth")

    batch_size = 32
    num_epochs = 20
    lr = 1e-3
    img_size = 128

    val_actor_ids = {21, 22, 23, 24}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    all_samples, class_names, class_to_idx, actor_counter = collect_samples_by_actor(data_dir)

    print("Classes:", class_names)
    print("Class to idx:", class_to_idx)
    print("Total images:", len(all_samples))
    print("Actors found:", sorted(actor_counter.keys()))
    print("Validation actors:", sorted(val_actor_ids))

    train_samples, val_samples = split_by_actor(all_samples, val_actor_ids)

    print("Train images:", len(train_samples))
    print("Val images:", len(val_samples))

    train_dataset = RAVDESSFrameDataset(train_samples, class_to_idx, transform=train_transform)
    val_dataset = RAVDESSFrameDataset(val_samples, class_to_idx, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = VGGStyleCNN128(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_uar, per_class_recalls = evaluate(model, val_loader, criterion, device, num_classes=len(class_names))

        scheduler.step(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val UAR: {val_uar:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}.")

    print(f"Best Val Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
