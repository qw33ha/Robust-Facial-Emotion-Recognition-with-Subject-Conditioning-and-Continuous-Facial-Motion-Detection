"""
evaluation/eval_ravdess_cnn_lstm_e2e.py

Purpose:
    Offline batch evaluation for the END-TO-END CNN-LSTM checkpoint
    produced by training/train_ravdess_cnn_lstm_e2e.py (no warm-start,
    no frozen encoder). Runs one clip-level inference per video on the
    actor-independent held-out split (actors 21-24) and reports
    overall/per-class metrics plus the confusion matrix.

    Architecturally identical to evaluation/eval_ravdess_cnn_lstm.py;
    the only differences are the checkpoint path, the output PNG name,
    and the plot title, so the two scripts' numbers are directly
    comparable (same model, same split, same sampler).

Pipeline:
    - Load weights/best_ravdess_cnn_lstm_e2e.pth into CNNLSTM128.
    - Walk ./output_gray_frames/<emotion>/ and group frames by video
      prefix "NN-NN-...-NN_actor_YY_<emotion>", keeping only videos
      whose actor_id is in VAL_ACTOR_IDS = {21, 22, 23, 24}.
    - For each video, uniformly sample SEQ_LEN frames via
      torch.linspace, build a [T, 1, 128, 128] clip, and run one
      forward pass.
    - Collect overall accuracy, per-class accuracy, prediction
      distribution, and a 7x7 confusion matrix (counts +
      row-normalized percentages).
    - Save the confusion matrix to
      outputs/ravdess_cnn_lstm_e2e_confusion_matrix.png.

Input:
    weights/best_ravdess_cnn_lstm_e2e.pth   Checkpoint produced by
                                            training/train_ravdess_cnn_lstm_e2e.py.
    ./output_gray_frames/<emotion>/         Frames grouped by emotion.

Output:
    Printed metrics on stdout.
    outputs/ravdess_cnn_lstm_e2e_confusion_matrix.png    Saved confusion matrix.

Usage (run from project root):
    python evaluation/eval_ravdess_cnn_lstm_e2e.py

Notes:
    - The per-epoch val confusion from training is saved separately
      to outputs/ravdess_cnn_lstm_e2e_val_confusion.png; this eval
      script is the canonical standalone reproduction.
    - CLASS_NAMES follow alphabetical sorted() order to match training.
    - SEQ_LEN must match the value used at training time.
"""

import re
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms


# =========================================================
# 1. Config
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(PROJECT_ROOT / "weights" / "best_ravdess_cnn_lstm_e2e.pth")
TEST_DIR = str(PROJECT_ROOT / "output_gray_frames")
CONFUSION_MATRIX_PATH = str(OUTPUTS_DIR / "ravdess_cnn_lstm_e2e_confusion_matrix.png")

# Actor-independent held-out split. Must match the val_actor_ids used in
# training/train_ravdess_cnn_lstm_e2e.py (actors 21-24).
VAL_ACTOR_IDS = {21, 22, 23, 24}

IMG_SIZE = 128
SEQ_LEN = 8
FEAT_DIM = 256
HIDDEN_DIM = 256
BIDIRECTIONAL = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alphabetical order — must match training's sorted() class discovery
# in training/train_ravdess_cnn_lstm_e2e.py so label indices align.
CLASS_NAMES = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# 2. Model (mirrors training/train_ravdess_cnn_lstm_e2e.py)
# =========================================================

class VGGStyleCNN128Encoder(nn.Module):
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


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)


class CNNLSTM128(nn.Module):
    def __init__(
        self,
        num_classes=7,
        in_channels=1,
        feat_dim=256,
        hidden_dim=256,
        num_layers=1,
        bidirectional=True,
        dropout=0.3,
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
        x = x.view(b * t, c, h, w)
        feats = self.encoder(x)
        feats = feats.view(b, t, -1)
        seq_out, _ = self.lstm(feats)
        pooled = self.pool(seq_out)
        return self.classifier(pooled)


# =========================================================
# 3. Transform
# =========================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# =========================================================
# 4. Load model
# =========================================================

def load_model():
    model = CNNLSTM128(
        num_classes=len(CLASS_NAMES),
        in_channels=1,
        feat_dim=FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        bidirectional=BIDIRECTIONAL,
        dropout=0.3,
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================================================
# 5. Collect video-level test samples
# =========================================================

def parse_video_id(filename: str) -> str:
    stem = Path(filename).stem
    if "_frame_" in stem:
        return stem.split("_frame_")[0]
    return stem


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


def collect_video_samples(test_dir: str, actor_ids):
    """
    Returns a list of (sorted_frame_paths, class_idx) tuples, one per
    RAVDESS video. Only videos whose actor_id is in `actor_ids` are kept.
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Folder not found: {test_dir}")

    grouped = defaultdict(list)
    skipped = 0

    for class_name in CLASS_NAMES:
        class_dir = test_path / class_name
        if not class_dir.exists():
            print(f"[WARN] Missing class folder: {class_dir}")
            continue

        for img_path in sorted(class_dir.iterdir()):
            if not (img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS):
                continue

            try:
                actor_id = parse_actor_id(img_path.name)
            except ValueError:
                skipped += 1
                continue

            if actor_id not in actor_ids:
                continue

            video_id = parse_video_id(img_path.name)
            grouped[(class_name, video_id, actor_id)].append(img_path)

    video_samples = []
    actor_video_counter = Counter()
    for (class_name, _video_id, actor_id), frame_paths in grouped.items():
        frame_paths = sorted(frame_paths)
        video_samples.append((frame_paths, CLASS_TO_IDX[class_name]))
        actor_video_counter[actor_id] += 1

    if not video_samples:
        raise RuntimeError(
            f"No video samples found in {test_dir} with actor_ids={sorted(actor_ids)}"
        )

    print(f"Filtered by actor_ids={sorted(actor_ids)}")
    print(f"  Videos per actor: {dict(sorted(actor_video_counter.items()))}")
    if skipped > 0:
        print(f"  Skipped (unparseable filename): {skipped}")

    return video_samples


# =========================================================
# 6. Sampling + evaluation
# =========================================================

def sample_clip(frame_paths, seq_len):
    n = len(frame_paths)
    if n == 0:
        raise ValueError("Empty frame list.")

    if n >= seq_len:
        indices = torch.linspace(0, n - 1, steps=seq_len).long().tolist()
        selected = [frame_paths[i] for i in indices]
    else:
        selected = list(frame_paths)
        while len(selected) < seq_len:
            selected.append(frame_paths[-1])
    return selected


def load_clip_tensor(frame_paths, seq_len):
    selected = sample_clip(frame_paths, seq_len)
    frames = []
    for p in selected:
        image = Image.open(p).convert("L")
        frames.append(transform(image))
    return torch.stack(frames, dim=0)  # [T, 1, H, W]


def evaluate(model, video_samples):
    correct = 0
    total = 0

    per_class_correct = {name: 0 for name in CLASS_NAMES}
    per_class_total = {name: 0 for name in CLASS_NAMES}

    num_classes = len(CLASS_NAMES)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    pred_counter = Counter()

    with torch.no_grad():
        for frame_paths, true_label in video_samples:
            clip = load_clip_tensor(frame_paths, SEQ_LEN)     # [T, 1, H, W]
            clip = clip.unsqueeze(0).to(DEVICE)               # [1, T, 1, H, W]

            logits = model(clip)
            pred_label = int(torch.argmax(logits, dim=1).item())

            total += 1
            true_name = CLASS_NAMES[true_label]
            pred_name = CLASS_NAMES[pred_label]

            per_class_total[true_name] += 1
            pred_counter[pred_name] += 1
            confusion[true_label, pred_label] += 1

            if pred_label == true_label:
                correct += 1
                per_class_correct[true_name] += 1

    accuracy = correct / total if total > 0 else 0.0
    return (
        accuracy,
        correct,
        total,
        per_class_correct,
        per_class_total,
        confusion,
        pred_counter,
    )


# =========================================================
# 7. Reporting
# =========================================================

def print_confusion_matrix(confusion, class_names):
    print("===== Confusion Matrix (counts) =====")
    header = "true\\pred".ljust(12) + "".join(name[:10].rjust(10) for name in class_names)
    print(header)
    for i, true_name in enumerate(class_names):
        row_str = true_name[:10].ljust(12)
        for j in range(len(class_names)):
            row_str += str(confusion[i, j]).rjust(10)
        print(row_str)
    print()


def print_confusion_matrix_percent(confusion, class_names):
    print("===== Confusion Matrix (row-normalized %) =====")
    header = "true\\pred".ljust(12) + "".join(name[:10].rjust(10) for name in class_names)
    print(header)
    for i, true_name in enumerate(class_names):
        row_sum = confusion[i].sum()
        row_str = true_name[:10].ljust(12)
        for j in range(len(class_names)):
            value = 100.0 * confusion[i, j] / row_sum if row_sum > 0 else 0.0
            row_str += f"{value:9.1f}%"
        print(row_str)
    print()


def print_prediction_distribution(pred_counter, total, class_names):
    print("===== Prediction Distribution =====")
    for class_name in class_names:
        count = pred_counter[class_name]
        ratio = count / total if total > 0 else 0.0
        print(f"{class_name:10s} : {count:5d} ({ratio * 100:.2f}%)")
    print()


def save_confusion_matrix_png(confusion, class_names, output_path,
                              title="RAVDESS CNN-LSTM (end-to-end) Confusion Matrix"):
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
# 8. Main
# =========================================================

def main():
    print("Using device:", DEVICE)
    print("Loading model from:", MODEL_PATH)
    print(f"Eval split: ACTOR-HELD-OUT (actors {sorted(VAL_ACTOR_IDS)})")

    model = load_model()
    video_samples = collect_video_samples(TEST_DIR, actor_ids=VAL_ACTOR_IDS)

    print(f"Found {len(video_samples)} test videos.\n")

    (
        accuracy,
        correct,
        total,
        per_class_correct,
        per_class_total,
        confusion,
        pred_counter,
    ) = evaluate(model, video_samples)

    print("===== Overall Accuracy =====")
    print(f"Correct:  {correct}")
    print(f"Total:    {total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

    print("===== Per-Class Accuracy =====")
    for class_name in CLASS_NAMES:
        cls_total = per_class_total[class_name]
        cls_correct = per_class_correct[class_name]
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        print(
            f"{class_name:10s} : {cls_correct:4d}/{cls_total:4d} "
            f"= {cls_acc:.4f} ({cls_acc * 100:.2f}%)"
        )
    print()

    uar = sum(
        (per_class_correct[name] / per_class_total[name])
        for name in CLASS_NAMES if per_class_total[name] > 0
    ) / len(CLASS_NAMES)
    print(f"UAR (unweighted average recall): {uar:.4f} ({uar * 100:.2f}%)\n")

    print_prediction_distribution(pred_counter, total, CLASS_NAMES)
    print_confusion_matrix(confusion, CLASS_NAMES)
    print_confusion_matrix_percent(confusion, CLASS_NAMES)

    save_confusion_matrix_png(
        confusion,
        CLASS_NAMES,
        CONFUSION_MATRIX_PATH,
        title="RAVDESS CNN-LSTM (end-to-end) Confusion Matrix — Held-out Actors 21-24",
    )
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
