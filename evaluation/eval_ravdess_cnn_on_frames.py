"""
evaluation/eval_ravdess_cnn_on_frames.py

Purpose:
    Offline batch evaluation: load the RAVDESS-trained VGGStyleCNN128
    checkpoint and run per-image inference on the actor-independent
    held-out split (actors 21-24), reporting overall/per-class metrics
    and saving the confusion matrix as a PNG.

Pipeline:
    - Load weights/best_ravdess_actor_split_model.pth into VGGStyleCNN128.
    - Walk ./output_gray_frames/<emotion>/ and enumerate images whose
      actor_id falls in VAL_ACTOR_IDS = {21, 22, 23, 24}, matching the
      val split used in training/train_ravdess_cnn_actor_split.py.
    - Run inference image by image; collect:
        * Overall accuracy
        * Per-class accuracy
        * Prediction distribution
        * Confusion matrix (raw counts + row-normalized percentages)
    - Save the confusion matrix to
      outputs/ravdess_cnn_confusion_matrix.png.

Input:
    weights/best_ravdess_actor_split_model.pth   Checkpoint produced by
                                                 training/train_ravdess_cnn_actor_split.py.
    ./output_gray_frames/<emotion>/              Test images.

Output:
    Printed metrics on stdout.
    outputs/ravdess_cnn_confusion_matrix.png     Saved confusion matrix.

Usage (run from project root):
    python evaluation/eval_ravdess_cnn_on_frames.py
"""

import re
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Config
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(PROJECT_ROOT / "weights" / "best_ravdess_actor_split_model.pth")
TEST_DIR = str(PROJECT_ROOT / "output_gray_frames")
CONFUSION_MATRIX_PATH = str(OUTPUTS_DIR / "ravdess_cnn_confusion_matrix.png")

# Actor-independent held-out split. Must match the val_actor_ids used in
# training/train_ravdess_cnn_actor_split.py (actors 21-24).
VAL_ACTOR_IDS = {21, 22, 23, 24}

IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alphabetical order — must match the training script's
# `sorted([p.name for p in data_path.iterdir() ...])` in
# training/train_ravdess_cnn_actor_split.py so label indices align.
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
# 2. Model
# =========================================================

class VGGStyleCNN128(nn.Module):
    """
    Deeper VGG-style CNN used for RAVDESS grayscale frames. See
    training/train_ravdess_cnn_actor_split.py for the definitive version.
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
    model = VGGStyleCNN128(num_classes=len(CLASS_NAMES)).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================================================
# 5. Collect test images
# =========================================================

def parse_actor_id(filename: str) -> int:
    """
    Mirror of training/train_ravdess_cnn_actor_split.py:parse_actor_id.

    RAVDESS frame filenames look like:
        "<prefix>_actor_<YY>_<emotion>_frame_<K>.png"
    or fall back to the raw RAVDESS filename:
        "03-01-05-01-02-01-12"  (actor id = 7th "-"-separated field)
    """
    stem = Path(filename).stem

    m = re.search(r"_actor_(\d+)_", stem)
    if m:
        return int(m.group(1))

    base = stem.split("_frame_")[0]
    parts = base.split("-")
    if len(parts) >= 7:
        return int(parts[6])

    raise ValueError(f"Cannot parse actor id from filename: {filename}")


def collect_test_samples(test_dir: str, actor_ids):
    """
    Walk output_gray_frames/<emotion>/ and keep only frames whose
    actor_id is in `actor_ids`.
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Folder not found: {test_dir}")

    samples = []
    kept_actors = Counter()
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

            kept_actors[actor_id] += 1
            samples.append((img_path, CLASS_TO_IDX[class_name]))

    if not samples:
        raise RuntimeError(
            f"No test images found in {test_dir} with actor_ids={sorted(actor_ids)}"
        )

    print(f"Filtered by actor_ids={sorted(actor_ids)}")
    print(f"  Kept per actor: {dict(sorted(kept_actors.items()))}")
    if skipped > 0:
        print(f"  Skipped (unparseable filename): {skipped}")

    return samples


# =========================================================
# 6. Evaluate accuracy
# =========================================================

def evaluate(model, samples):
    correct = 0
    total = 0

    per_class_correct = {name: 0 for name in CLASS_NAMES}
    per_class_total = {name: 0 for name in CLASS_NAMES}

    num_classes = len(CLASS_NAMES)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    pred_counter = Counter()

    with torch.no_grad():
        for img_path, true_label in samples:
            image = Image.open(img_path).convert("L")
            x = transform(image).unsqueeze(0).to(DEVICE)

            logits = model(x)
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
                              title="RAVDESS VGGStyleCNN128 Confusion Matrix"):
    """Save a matplotlib heatmap of the confusion matrix to output_path."""
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
                j, i, format(confusion[i, j], "d"),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# 7. Main
# =========================================================

def main():
    print("Using device:", DEVICE)
    print("Loading model from:", MODEL_PATH)
    print(f"Eval split: ACTOR-HELD-OUT (actors {sorted(VAL_ACTOR_IDS)})")

    model = load_model()
    samples = collect_test_samples(TEST_DIR, actor_ids=VAL_ACTOR_IDS)

    print(f"Found {len(samples)} test images.\n")

    (
        accuracy,
        correct,
        total,
        per_class_correct,
        per_class_total,
        confusion,
        pred_counter,
    ) = evaluate(model, samples)

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

    print_prediction_distribution(pred_counter, total, CLASS_NAMES)
    print_confusion_matrix(confusion, CLASS_NAMES)
    print_confusion_matrix_percent(confusion, CLASS_NAMES)

    save_confusion_matrix_png(
        confusion,
        CLASS_NAMES,
        CONFUSION_MATRIX_PATH,
        title="RAVDESS VGGStyleCNN128 Confusion Matrix — Held-out Actors 21-24",
    )
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
