"""
training/train_ravdess_cnn_lstm.py

Purpose:
    Train a CNN-LSTM on RAVDESS grayscale frame clips using an actor-
    independent (subject-independent) train/val split. The per-frame
    encoder is warm-started from the frame-level checkpoint
    (weights/best_ravdess_actor_split_model.pth) and frozen, so only
    the temporal aggregator (BiLSTM + attention pool + MLP head) is
    learned.

    For the end-to-end variant (encoder trained from scratch together
    with the temporal head, no warm-start / no freeze), see
    training/train_ravdess_cnn_lstm_e2e.py.

Pipeline:
    - Load grayscale frames from ./output_gray_frames/, group by video
      using the per-file prefix "NN-NN-...-NN_actor_YY_<emotion>" and
      by actor id parsed from the filename.
    - Use actors 21/22/23/24 as the validation set; all other actors
      go into training.
    - Sample seq_len frames uniformly from each video via torch.linspace;
      if a video has fewer than seq_len frames, repeat the last frame.
    - Encoder: VGGStyleCNN128 features (+ adaptive pool + projection to
      feat_dim). Warm-started from the frame checkpoint and frozen.
    - Temporal aggregator: BiLSTM (num_layers=1, hidden=256) + attention
      pooling over the time axis.
    - Head: 2-layer MLP with dropout.
    - 20 epochs, Adam lr=1e-4 + weight decay 1e-4, ReduceLROnPlateau on
      val accuracy.
    - On each new best val_acc, save both the checkpoint and the
      confusion matrix PNG of that validation pass.

Input:
    ./output_gray_frames/<emotion>/*.png
    weights/best_ravdess_actor_split_model.pth   Warm-start source.

Output:
    weights/best_ravdess_cnn_lstm.pth               Best checkpoint.
    outputs/ravdess_cnn_lstm_val_confusion.png      Best-epoch confusion.

Usage (run from project root):
    python training/train_ravdess_cnn_lstm.py

Notes:
    - Frame filename format expected:
      "NN-NN-...-NN_actor_YY_<emotion>_frame_NNNNN.png".
    - Class labels follow alphabetical sorted() order so inference-time
      indices stay consistent with training/train_ravdess_cnn_actor_split.py
      and evaluation/eval_ravdess_cnn_on_frames.py.
"""

import re
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# =========================================================
# 1. Model
# =========================================================

class VGGStyleCNN128Encoder(nn.Module):
    """
    Encoder-only version of VGGStyleCNN128 (features + adaptive pool +
    linear projection). Matches the convolutional stack of
    training/train_ravdess_cnn_actor_split.py::VGGStyleCNN128, so the
    `features.*` weights of best_ravdess_actor_split_model.pth load
    directly here.
    """

    def __init__(self, in_channels=1, feat_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=0),
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
        self.proj = nn.Linear(512 * 2 * 2, feat_dim)

    def forward(self, x):
        # x: [N, 1, H, W]
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x

    def load_features_from_frame_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load only the `features.*` weights from a VGGStyleCNN128
        frame-level checkpoint. `proj` stays randomly initialized.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        own_state = self.state_dict()

        loaded, skipped = 0, 0
        for name, param in state_dict.items():
            if name.startswith("features.") and name in own_state:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        self.load_state_dict(own_state, strict=strict)
        return loaded, skipped


class AttentionPool(nn.Module):
    """
    Learned attention pooling over the temporal axis.
    Input  [B, T, D] -> Output [B, D].
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        scores = self.attn(x)                 # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)     # [B, D]
        return pooled


class CNNLSTM128(nn.Module):
    """
    Per-frame VGGStyleCNN128Encoder + BiLSTM + attention pool + MLP head.

    Forward input:  [B, T, C, H, W]
    Forward output: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        feat_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = VGGStyleCNN128Encoder(in_channels=in_channels, feat_dim=feat_dim)

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        temporal_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.pool = AttentionPool(temporal_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)        # [B*T, C, H, W]
        feats = self.encoder(x)           # [B*T, feat_dim]
        feats = feats.view(b, t, -1)      # [B, T, feat_dim]

        seq_out, _ = self.lstm(feats)     # [B, T, temporal_dim]
        pooled = self.pool(seq_out)       # [B, temporal_dim]
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits

    def set_encoder_frozen(self, frozen: bool):
        """Freeze or unfreeze the encoder (conv + proj)."""
        for p in self.encoder.parameters():
            p.requires_grad = not frozen
        # When frozen, putting encoder in eval mode also freezes BN stats.
        if frozen:
            self.encoder.eval()


# =========================================================
# 2. Filename parsing
# =========================================================

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


def parse_video_id(filename: str) -> str:
    """
    Video id = everything before the "_frame_NNNNN" suffix. Uniquely
    identifies a RAVDESS clip across frames extracted from it.
    """
    stem = Path(filename).stem
    if "_frame_" in stem:
        return stem.split("_frame_")[0]
    return stem


# =========================================================
# 3. Dataset
# =========================================================

class RAVDESSSequenceDataset(Dataset):
    """
    One item = one video = a tensor [T, C, H, W] of seq_len uniformly
    sampled frames + emotion label.
    """

    def __init__(self, video_samples, class_to_idx, transform=None, seq_len=8):
        self.video_samples = video_samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.video_samples)

    def _sample_frames(self, frame_paths):
        n = len(frame_paths)
        if n == 0:
            raise ValueError("Empty frame list for one video sample.")

        if n >= self.seq_len:
            indices = torch.linspace(0, n - 1, steps=self.seq_len).long().tolist()
            selected = [frame_paths[i] for i in indices]
        else:
            selected = list(frame_paths)
            while len(selected) < self.seq_len:
                selected.append(frame_paths[-1])
        return selected

    def __getitem__(self, idx):
        frame_paths, label_name = self.video_samples[idx]
        selected_paths = self._sample_frames(frame_paths)

        frames = []
        for img_path in selected_paths:
            image = Image.open(img_path).convert("L")
            if self.transform is not None:
                image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]
        label = self.class_to_idx[label_name]
        return frames, label


def collect_video_samples_by_actor(data_dir: str):
    data_path = Path(data_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    class_names = sorted([p.name for p in data_path.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    grouped = defaultdict(list)
    actor_counter = Counter()

    for class_name in class_names:
        class_dir = data_path / class_name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in image_exts:
                actor_id = parse_actor_id(img_path.name)
                video_id = parse_video_id(img_path.name)
                grouped[(class_name, video_id, actor_id)].append(img_path)

    all_video_samples = []
    for (class_name, _video_id, actor_id), frame_paths in grouped.items():
        frame_paths = sorted(frame_paths)
        all_video_samples.append((frame_paths, class_name, actor_id))
        actor_counter[actor_id] += 1

    return all_video_samples, class_names, class_to_idx, actor_counter


def split_by_actor(all_video_samples, val_actor_ids):
    train_samples, val_samples = [], []
    for frame_paths, class_name, actor_id in all_video_samples:
        item = (frame_paths, class_name)
        if actor_id in val_actor_ids:
            val_samples.append(item)
        else:
            train_samples.append(item)
    return train_samples, val_samples


# =========================================================
# 4. Metrics / plotting
# =========================================================

def compute_uar_from_confusion(confusion, num_classes):
    recalls = []
    for i in range(num_classes):
        tp = confusion[i, i]
        fn = confusion[i].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    uar = sum(recalls) / num_classes if num_classes > 0 else 0.0
    return uar, recalls


def save_confusion_matrix_png(
    confusion,
    class_names,
    output_path,
    title="RAVDESS CNN-LSTM Val Confusion Matrix",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = confusion.max() / 2.0 if confusion.max() > 0 else 0.5
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j, i, format(int(confusion[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 5. Train / evaluate
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    # model.train() sets everything to train mode; the encoder is frozen
    # in this script, so we re-pin it to eval to keep BN running stats
    # pinned to the warm-started values.
    model.train()
    model.encoder.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for videos, labels in tqdm(loader, desc="Train", leave=False):
        videos = videos.to(device)   # [B, T, C, H, W]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * videos.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for videos, labels in tqdm(loader, desc="Val", leave=False):
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * videos.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for t, p in zip(labels.cpu(), preds.cpu()):
            confusion[t.long(), p.long()] += 1

    val_loss = total_loss / total_samples
    val_acc = total_correct / total_samples

    confusion = confusion.numpy()
    val_uar, per_class_recalls = compute_uar_from_confusion(confusion, num_classes)

    return val_loss, val_acc, val_uar, per_class_recalls, confusion


# =========================================================
# 6. Main
# =========================================================

def main():
    # ---------- paths ----------
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = str(PROJECT_ROOT / "output_gray_frames")
    frame_ckpt_path = str(WEIGHTS_DIR / "best_ravdess_actor_split_model.pth")
    best_model_path = str(WEIGHTS_DIR / "best_ravdess_cnn_lstm.pth")
    best_confusion_path = str(OUTPUTS_DIR / "ravdess_cnn_lstm_val_confusion.png")

    # ---------- hyperparameters ----------
    batch_size = 8
    num_epochs = 20
    lr = 1e-4
    img_size = 128
    seq_len = 8
    feat_dim = 256
    hidden_dim = 256

    val_actor_ids = {21, 22, 23, 24}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- transforms ----------
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

    # ---------- data ----------
    all_video_samples, class_names, class_to_idx, actor_counter = (
        collect_video_samples_by_actor(data_dir)
    )

    print("Classes:", class_names)
    print("Class to idx:", class_to_idx)
    print("Total videos:", len(all_video_samples))
    print("Actors found:", sorted(actor_counter.keys()))
    print("Validation actors:", sorted(val_actor_ids))

    train_samples, val_samples = split_by_actor(all_video_samples, val_actor_ids)
    print("Train videos:", len(train_samples))
    print("Val videos:", len(val_samples))

    train_dataset = RAVDESSSequenceDataset(
        train_samples, class_to_idx, transform=train_transform, seq_len=seq_len
    )
    val_dataset = RAVDESSSequenceDataset(
        val_samples, class_to_idx, transform=val_transform, seq_len=seq_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---------- model ----------
    model = CNNLSTM128(
        num_classes=len(class_names),
        in_channels=1,
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        bidirectional=True,
        dropout=0.3,
    ).to(device)

    if not Path(frame_ckpt_path).exists():
        raise FileNotFoundError(
            f"Warm-start checkpoint not found: {frame_ckpt_path}. "
            "Run training/train_ravdess_cnn_actor_split.py first, or use "
            "training/train_ravdess_cnn_lstm_e2e.py for an end-to-end "
            "variant that trains the encoder from scratch."
        )

    loaded, skipped = model.encoder.load_features_from_frame_checkpoint(frame_ckpt_path)
    print(f"Warm-started encoder from {frame_ckpt_path}: "
          f"{loaded} tensors loaded, {skipped} skipped.")

    model.set_encoder_frozen(True)
    print("Encoder frozen: True")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    # ---------- optimizer / scheduler ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # ---------- train loop ----------
    best_acc = 0.0
    best_uar = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, val_uar, per_class_recalls, val_confusion = evaluate(
            model, val_loader, criterion, device, num_classes=len(class_names)
        )

        scheduler.step(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val UAR: {val_uar:.4f}")
        print("  Per-class recall: " + ", ".join(
            f"{name}={r:.3f}" for name, r in zip(class_names, per_class_recalls)
        ))

        if val_acc > best_acc:
            best_acc = val_acc
            best_uar = val_uar
            torch.save(model.state_dict(), best_model_path)
            save_confusion_matrix_png(val_confusion, class_names, best_confusion_path)
            print(f"Saved best model to {best_model_path}.")
            print(f"Saved best-epoch confusion to {best_confusion_path}.")

    print(f"Best Val Accuracy: {best_acc:.4f} | Val UAR at best: {best_uar:.4f}")


if __name__ == "__main__":
    main()
