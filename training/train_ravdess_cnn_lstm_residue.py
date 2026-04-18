"""
training/train_ravdess_cnn_lstm_residue.py

Purpose:
    CNN-LSTM with SUBJECT CONDITIONING via feature-level RESIDUE.

    For each clip, the encoder feature of every frame is
    subject-normalized by subtracting a per-actor "neutral template":
    the mean encoder feature of K randomly sampled neutral frames
    from the same actor. The resulting residue (f_t - f_neutral) is
    fed into the temporal aggregator (BiLSTM + attention pool + MLP
    head). The intent is to factor identity / face shape out of the
    per-frame embedding so that the BiLSTM sees only the
    expression-induced component.

    The rest of the recipe is identical to
    training/train_ravdess_cnn_lstm.py: same warm-start (encoder
    loaded from best_ravdess_actor_split_model.pth and frozen), same
    actor-independent split (actors 21-24 = val), same optimiser and
    epochs. This makes the val-accuracy delta between this script
    and train_ravdess_cnn_lstm.py a clean attribution of the residue
    trick.

Pipeline:
    - Load grayscale frames from ./output_gray_frames/.
    - Precompute neutral features per actor ONCE with the frozen
      encoder: for every actor, encode all of its neutral frames
      (using the non-augmenting val_transform) into a tensor
      [N_actor, feat_dim]. Also cache the mean as the canonical
      neutral template used at validation time.
    - Training step, per clip in the batch:
        * Randomly sample NEUTRAL_K indices from that actor's
          neutral feature bank, take their mean -> f_neutral [D].
        * Encode the clip's seq_len frames -> [T, D].
        * residue_t = f_t - f_neutral, broadcast over T.
        * BiLSTM + attention pool + MLP -> logits.
    - Validation step: same as training but uses the precomputed
      per-actor MEAN of all neutral frames as a deterministic
      template (no random sampling).
    - 20 epochs, Adam lr=1e-4 + weight decay 1e-4,
      ReduceLROnPlateau on val accuracy. Save the best-by-val-acc
      checkpoint and the corresponding confusion matrix PNG.

Input:
    ./output_gray_frames/<emotion>/*.png
    weights/best_ravdess_actor_split_model.pth   Warm-start source.

Output:
    weights/best_ravdess_cnn_lstm_residue.pth
    outputs/ravdess_cnn_lstm_residue_val_confusion.png

Usage (run from project root):
    python training/train_ravdess_cnn_lstm_residue.py

Notes:
    - Neutral frames of validation actors (21-24) ARE used to build
      their neutral templates. This is intentional: in identity-aware
      FER, only the non-neutral expression *labels* of held-out
      actors are unseen. Neutral is treated as enrollment data, i.e.
      one still needs to see the person's resting face to subject-
      normalize them. No validation-side expression label leaks.
    - NEUTRAL_K defaults to 8. If an actor has fewer than NEUTRAL_K
      neutral frames, indices are drawn with replacement.
    - Class indices follow alphabetical sorted() order so they stay
      consistent with train_ravdess_cnn_actor_split.py,
      train_ravdess_cnn_lstm.py, and the eval scripts.
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
    """Learned attention pooling over the temporal axis: [B,T,D] -> [B,D]."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)


class CNNLSTMResidue128(nn.Module):
    """
    VGGStyleCNN128Encoder + subject-residue + BiLSTM + attention pool
    + MLP head.

    forward(x, neutral_feat) where:
      x:           [B, T, C, H, W]
      neutral_feat [B, feat_dim]    per-sample actor neutral template.
    output:        [B, num_classes]

    Note that feat_dim stays the same as the baseline CNN-LSTM — the
    residue is computed *in place* (f_t - f_neutral) rather than
    concatenated — so the rest of the temporal head is literally the
    same shape as in train_ravdess_cnn_lstm.py.
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

    def forward(self, x, neutral_feat):
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)
        feats = self.encoder(x)                          # [B*T, D]
        feats = feats.view(b, t, -1)                     # [B, T, D]

        # Subject residue: broadcast the actor neutral over time.
        residue = feats - neutral_feat.unsqueeze(1)      # [B, T, D]

        seq_out, _ = self.lstm(residue)                  # [B, T, 2H]
        pooled = self.pool(seq_out)                      # [B, 2H]
        logits = self.classifier(pooled)                 # [B, K]
        return logits

    def set_encoder_frozen(self, frozen: bool):
        for p in self.encoder.parameters():
            p.requires_grad = not frozen
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
    stem = Path(filename).stem
    if "_frame_" in stem:
        return stem.split("_frame_")[0]
    return stem


# =========================================================
# 3. Dataset
# =========================================================

class RAVDESSSequenceDatasetWithActor(Dataset):
    """
    One item = one video = a tensor [T, C, H, W] of seq_len uniformly
    sampled frames + emotion label + actor id. The actor id is needed
    so the train loop can look up the per-actor neutral bank.
    """

    def __init__(self, video_samples, class_to_idx, transform=None, seq_len=8):
        # video_samples: list of (sorted_frame_paths, class_name, actor_id)
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
        frame_paths, label_name, actor_id = self.video_samples[idx]
        selected_paths = self._sample_frames(frame_paths)

        frames = []
        for img_path in selected_paths:
            image = Image.open(img_path).convert("L")
            if self.transform is not None:
                image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]
        label = self.class_to_idx[label_name]
        return frames, label, int(actor_id)


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
        item = (frame_paths, class_name, actor_id)
        if actor_id in val_actor_ids:
            val_samples.append(item)
        else:
            train_samples.append(item)
    return train_samples, val_samples


# =========================================================
# 4. Per-actor neutral feature bank
# =========================================================

def collect_neutral_frame_paths_by_actor(data_dir: str):
    """Walk ./output_gray_frames/neutral/ and group frame paths by actor id."""
    neutral_dir = Path(data_dir) / "neutral"
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not neutral_dir.exists():
        raise FileNotFoundError(
            f"Neutral emotion folder not found: {neutral_dir}. "
            "Did you run preprocess/extract_ravdess_gray_frames.py?"
        )

    by_actor = defaultdict(list)
    for img_path in sorted(neutral_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in image_exts:
            try:
                actor_id = parse_actor_id(img_path.name)
            except ValueError:
                continue
            by_actor[actor_id].append(img_path)
    return by_actor


@torch.no_grad()
def build_actor_neutral_bank(
    encoder: nn.Module,
    neutral_paths_by_actor,
    transform,
    device,
    batch_size: int = 64,
):
    """
    Encode every neutral frame through the frozen encoder, grouped by
    actor. Returns:
        feats_by_actor: Dict[int, Tensor[N_actor, feat_dim]]   (on CPU)
        means_by_actor: Dict[int, Tensor[feat_dim]]            (on CPU)
    """
    encoder.eval()

    feats_by_actor = {}
    means_by_actor = {}

    for actor_id, paths in sorted(neutral_paths_by_actor.items()):
        buf = []
        feats = []
        for p in paths:
            image = Image.open(p).convert("L")
            buf.append(transform(image))
            if len(buf) >= batch_size:
                batch = torch.stack(buf, dim=0).to(device)
                feats.append(encoder(batch).detach().cpu())
                buf = []
        if buf:
            batch = torch.stack(buf, dim=0).to(device)
            feats.append(encoder(batch).detach().cpu())

        if len(feats) == 0:
            continue

        bank = torch.cat(feats, dim=0)          # [N_actor, D]
        feats_by_actor[actor_id] = bank
        means_by_actor[actor_id] = bank.mean(dim=0)

    return feats_by_actor, means_by_actor


def sample_neutral_means(feats_by_actor, actor_ids, k: int, device):
    """
    For each actor in the batch, randomly sample k indices from its
    neutral feature bank and return the per-actor mean of those k rows.
    actor_ids: LongTensor / list[int] of length B.
    Returns tensor [B, feat_dim] on `device`.
    """
    out = []
    for aid in actor_ids:
        aid = int(aid)
        bank = feats_by_actor[aid]              # [N, D]
        n = bank.size(0)
        if n == 0:
            raise RuntimeError(f"No neutral frames for actor {aid}.")
        if n >= k:
            idx = torch.randperm(n)[:k]
        else:
            idx = torch.randint(0, n, (k,))
        out.append(bank[idx].mean(dim=0))
    return torch.stack(out, dim=0).to(device)


def lookup_neutral_means(means_by_actor, actor_ids, device):
    """
    Deterministic variant used at validation time: just fetches the
    precomputed per-actor mean of ALL neutral frames.
    """
    return torch.stack(
        [means_by_actor[int(a)] for a in actor_ids], dim=0
    ).to(device)


# =========================================================
# 5. Metrics / plotting
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
    title="RAVDESS CNN-LSTM (residue) Val Confusion Matrix",
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
# 6. Train / evaluate
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device,
                    feats_by_actor, neutral_k):
    # Trainable parts: LSTM + attention + MLP head. Encoder stays frozen
    # (incl. BN running stats) to keep the warm-started features stable.
    model.train()
    model.encoder.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for videos, labels, actor_ids in tqdm(loader, desc="Train", leave=False):
        videos = videos.to(device)                       # [B, T, C, H, W]
        labels = labels.to(device)

        neutral_means = sample_neutral_means(
            feats_by_actor, actor_ids.tolist(), k=neutral_k, device=device,
        )                                                # [B, D]

        optimizer.zero_grad()
        outputs = model(videos, neutral_means)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * videos.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, means_by_actor):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for videos, labels, actor_ids in tqdm(loader, desc="Val", leave=False):
        videos = videos.to(device)
        labels = labels.to(device)

        # Val uses the deterministic full-mean template per actor.
        neutral_means = lookup_neutral_means(
            means_by_actor, actor_ids.tolist(), device=device,
        )

        outputs = model(videos, neutral_means)
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
# 7. Main
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
    best_model_path = str(WEIGHTS_DIR / "best_ravdess_cnn_lstm_residue.pth")
    best_confusion_path = str(OUTPUTS_DIR / "ravdess_cnn_lstm_residue_val_confusion.png")

    # ---------- hyperparameters ----------
    batch_size = 8
    num_epochs = 20
    lr = 1e-4
    img_size = 128
    seq_len = 8
    feat_dim = 256
    hidden_dim = 256
    neutral_k = 8                         # K=8 as agreed

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
    # Neutral template uses the non-augmenting transform so the residue
    # subtracts a canonical "resting face" rather than a random crop /
    # flip of one.
    neutral_transform = val_transform

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

    train_dataset = RAVDESSSequenceDatasetWithActor(
        train_samples, class_to_idx, transform=train_transform, seq_len=seq_len
    )
    val_dataset = RAVDESSSequenceDatasetWithActor(
        val_samples, class_to_idx, transform=val_transform, seq_len=seq_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---------- model ----------
    model = CNNLSTMResidue128(
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
            "Run training/train_ravdess_cnn_actor_split.py first."
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

    # ---------- per-actor neutral feature bank ----------
    print("Building per-actor neutral feature bank ...")
    neutral_paths_by_actor = collect_neutral_frame_paths_by_actor(data_dir)
    print("  Actors with neutral frames:", sorted(neutral_paths_by_actor.keys()))
    for aid in sorted(neutral_paths_by_actor.keys()):
        print(f"    actor {aid:02d}: {len(neutral_paths_by_actor[aid])} neutral frames")

    feats_by_actor, means_by_actor = build_actor_neutral_bank(
        encoder=model.encoder,
        neutral_paths_by_actor=neutral_paths_by_actor,
        transform=neutral_transform,
        device=device,
        batch_size=64,
    )
    print(f"Neutral bank ready. feat_dim={feat_dim}, "
          f"actors covered={len(feats_by_actor)}, "
          f"sampling K={neutral_k} per clip during training.")

    # Sanity check: every train/val actor must have a neutral bank.
    all_actor_ids = set(a for _, _, a in all_video_samples)
    missing = all_actor_ids - set(feats_by_actor.keys())
    if missing:
        raise RuntimeError(
            f"Actors {sorted(missing)} have non-neutral frames but no "
            "neutral frames in ./output_gray_frames/neutral/. The "
            "residue trick requires at least one neutral frame per actor."
        )

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
            feats_by_actor=feats_by_actor,
            neutral_k=neutral_k,
        )
        val_loss, val_acc, val_uar, per_class_recalls, val_confusion = evaluate(
            model, val_loader, criterion, device,
            num_classes=len(class_names),
            means_by_actor=means_by_actor,
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
