"""
evaluation/eval_ravdess_clip_zeroshot.py

Purpose:
    Zero-shot evaluation: use HuggingFace CLIP (ViT-B/32) directly on
    RAVDESS videos for emotion recognition, with no training. Serves as
    a comparison baseline.

Pipeline:
    - Load openai/clip-vit-base-patch32.
    - Walk ./RAVDESS and collect every .mp4.
    - Uniformly sample 5 frames per video.
    - For each class, build text embeddings from 3 prompt templates and
      average them:
        * "a facial expression showing {} emotion"
        * "a person with a {} facial expression"
        * "a face expressing {}"
    - Compute cosine similarity between image and text embeddings,
      softmax to get per-frame class probabilities.
    - Average over 5 frames to get a video-level prediction.
    - Merge neutral and calm into neutral_or_calm (7 classes total).

Input:
    ./RAVDESS/                                  Original video directory.

Output:
    ravdess_confusion_matrix.png                Video-level confusion matrix.
    ravdess_clip_zero_shot_predictions.json     Per-video probability details.

Usage (run from project root):
    python evaluation/eval_ravdess_clip_zeroshot.py
"""

import os
import cv2
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt


# =========================================================
# 1. Config
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
RAVDESS_ROOT = "BASE_DIR/../RAVDESS"   # RAVDESS dataset root
VIDEO_EXTENSIONS = {".mp4"}

# Number of uniformly sampled frames per video.
FRAMES_PER_VIDEO = 5

# Random seed.
SEED = 42

# CLIP model name.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Evaluation labels: neutral and calm are merged into a single class.
MERGED_LABELS = [
    "neutral_or_calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

# Prompt templates used to build text embeddings for each class.
PROMPT_TEMPLATES = [
    "a facial expression showing {} emotion",
    "a person with a {} facial expression",
    "a face expressing {}",
]

# Whether to save detailed per-video predictions.
SAVE_PREDICTIONS_JSON = True
PREDICTIONS_JSON_PATH = "ravdess_clip_zero_shot_predictions.json"

# Path for the confusion matrix PNG.
CONFUSION_MATRIX_PNG = "ravdess_confusion_matrix.png"


# =========================================================
# 2. RAVDESS label parsing
# =========================================================

# RAVDESS emotion code:
# 01 neutral
# 02 calm
# 03 happy
# 04 sad
# 05 angry
# 06 fearful
# 07 disgust
# 08 surprised

RAW_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Mapping from raw label to merged label.
MERGED_EMOTION_MAP = {
    "neutral": "neutral_or_calm",
    "calm": "neutral_or_calm",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "disgust": "disgust",
    "surprised": "surprised",
}


def parse_ravdess_filename(filename: str):
    """
    Example:
    02-01-06-01-02-01-12.mp4

    Fields:
    Modality - Vocal channel - Emotion - Emotional intensity
    - Statement - Repetition - Actor

    We only need emotion and actor here.
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename format: {filename}")

    emotion_code = parts[2]
    actor_id = parts[6]

    raw_emotion = RAW_EMOTION_MAP.get(emotion_code)
    if raw_emotion is None:
        raise ValueError(f"Unknown emotion code {emotion_code} in {filename}")

    merged_emotion = MERGED_EMOTION_MAP[raw_emotion]

    return {
        "emotion_code": emotion_code,
        "raw_emotion": raw_emotion,
        "merged_emotion": merged_emotion,
        "actor_id": actor_id,
        "stem": stem,
    }


def find_all_videos(root_dir):
    video_paths = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                video_paths.append(os.path.join(root, fname))
    video_paths.sort()
    return video_paths


# =========================================================
# 3. Frame extraction (no cropping)
# =========================================================

def sample_frame_indices(total_frames: int, frames_per_video: int):
    """
    Uniformly sample frame indices across a video.
    Avoid first/last exact edges where possible.
    """
    if total_frames <= 0:
        return []

    if total_frames <= frames_per_video:
        return list(range(total_frames))

    indices = []
    for i in range(frames_per_video):
        # sample near center of each segment
        pos = (i + 0.5) * total_frames / frames_per_video
        idx = min(total_frames - 1, max(0, int(pos)))
        indices.append(idx)

    # remove duplicates while preserving order
    dedup = []
    seen = set()
    for x in indices:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def extract_sampled_frames(video_path: str, frames_per_video: int):
    """
    Return a list of PIL RGB images sampled uniformly from the video.
    No cropping is performed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, frames_per_video)

    if not frame_indices:
        cap.release()
        return []

    frames = []
    target_set = set(frame_indices)
    current_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append((current_idx, pil_img))

            if len(frames) == len(frame_indices):
                break

        current_idx += 1

    cap.release()

    # sort by sampled frame index
    frames.sort(key=lambda x: x[0])
    return frames


# =========================================================
# 4. CLIP zero-shot classifier
# =========================================================

class ZeroShotCLIPClassifier:
    def __init__(self, model_name: str, labels, prompt_templates, device=None):
        self.labels = labels
        self.prompt_templates = prompt_templates
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        print("Using device:", self.device)
        print("Model class:", type(self.model))
        print("Transformers CLIP loaded from:", self.model.__class__.__module__)

        self.text_embeds = self._build_text_embeddings()

    def _encode_text(self, prompts):
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            # Preferred path: official CLIP API, which should return text embeddings
            try:
                text_features = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            except Exception:
                text_features = None

            # Fallback path for environments that return a model output object instead
            if text_features is None or not isinstance(text_features, torch.Tensor):
                text_outputs = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                pooled = text_outputs.pooler_output
                text_features = self.model.text_projection(pooled)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _encode_image(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            try:
                image_features = self.model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )
            except Exception:
                image_features = None

            if image_features is None or not isinstance(image_features, torch.Tensor):
                vision_outputs = self.model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                pooled = vision_outputs.pooler_output
                image_features = self.model.visual_projection(pooled)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _build_text_embeddings(self):
        self.model.eval()
        all_label_embeds = []

        for label in self.labels:
            prompts = [
                tpl.format(label.replace("_", " "))
                for tpl in self.prompt_templates
            ]
            text_features = self._encode_text(prompts)
            avg_text_feature = text_features.mean(dim=0, keepdim=True)
            avg_text_feature = avg_text_feature / avg_text_feature.norm(dim=-1, keepdim=True)
            all_label_embeds.append(avg_text_feature)

        text_embeds = torch.cat(all_label_embeds, dim=0)
        return text_embeds

    def predict_image_probs(self, pil_image):
        self.model.eval()
        image_features = self._encode_image(pil_image)
        logits = 100.0 * image_features @ self.text_embeds.T
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.detach().cpu().numpy()

    def predict_video(self, frames):
        per_frame_probs = []
        used_frame_indices = []

        for frame_idx, pil_image in frames:
            probs = self.predict_image_probs(pil_image)
            per_frame_probs.append(probs)
            used_frame_indices.append(frame_idx)

        if not per_frame_probs:
            return None, None, None, used_frame_indices

        per_frame_probs = np.stack(per_frame_probs, axis=0)
        avg_probs = per_frame_probs.mean(axis=0)
        pred_idx = int(np.argmax(avg_probs))
        pred_label = self.labels[pred_idx]

        return pred_label, avg_probs, per_frame_probs, used_frame_indices

# =========================================================
# 5. Evaluation + plotting
# =========================================================

def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="RAVDESS Zero-Shot CLIP Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# =========================================================
# 6. Main
# =========================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("RAVDESS Zero-Shot CLIP Evaluation")
    print("=" * 70)
    print(f"RAVDESS_ROOT      : {RAVDESS_ROOT}")
    print(f"CLIP_MODEL_NAME   : {CLIP_MODEL_NAME}")
    print(f"FRAMES_PER_VIDEO  : {FRAMES_PER_VIDEO}")
    print(f"MERGED_LABELS     : {MERGED_LABELS}")
    print(f"PROMPT_TEMPLATES  : {PROMPT_TEMPLATES}")
    print("=" * 70)

    video_paths = find_all_videos(RAVDESS_ROOT)
    if not video_paths:
        raise RuntimeError(f"No videos found under {RAVDESS_ROOT}")

    print(f"Found {len(video_paths)} videos.")

    classifier = ZeroShotCLIPClassifier(
        model_name=CLIP_MODEL_NAME,
        labels=MERGED_LABELS,
        prompt_templates=PROMPT_TEMPLATES,
    )

    y_true = []
    y_pred = []
    detailed_results = []

    skipped = 0

    for video_path in tqdm(video_paths, desc="Evaluating videos"):
        fname = os.path.basename(video_path)

        try:
            meta = parse_ravdess_filename(fname)
        except Exception as e:
            print(f"[WARN] Skip malformed file {fname}: {e}")
            skipped += 1
            continue

        true_label = meta["merged_emotion"]

        frames = extract_sampled_frames(video_path, FRAMES_PER_VIDEO)
        if len(frames) == 0:
            print(f"[WARN] No frames extracted from: {video_path}")
            skipped += 1
            continue

        pred_label, avg_probs, per_frame_probs, used_frame_indices = classifier.predict_video(frames)
        if pred_label is None:
            print(f"[WARN] Prediction failed for: {video_path}")
            skipped += 1
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

        result_item = {
            "video_path": video_path,
            "filename": fname,
            "actor_id": meta["actor_id"],
            "raw_emotion": meta["raw_emotion"],
            "merged_true_label": true_label,
            "predicted_label": pred_label,
            "used_frame_indices": used_frame_indices,
            "avg_probs": {
                label: float(avg_probs[i]) for i, label in enumerate(MERGED_LABELS)
            },
        }
        detailed_results.append(result_item)

    print("\n" + "=" * 70)
    print("Evaluation finished")
    print("=" * 70)
    print(f"Total videos evaluated : {len(y_true)}")
    print(f"Skipped videos         : {skipped}")

    if len(y_true) == 0:
        raise RuntimeError("No valid samples were evaluated.")

    acc = accuracy_score(y_true, y_pred)
    print(f"\nVideo-level Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=MERGED_LABELS, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=MERGED_LABELS)
    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, MERGED_LABELS, CONFUSION_MATRIX_PNG)
    print(f"\nSaved confusion matrix to: {CONFUSION_MATRIX_PNG}")

    if SAVE_PREDICTIONS_JSON:
        with open(PREDICTIONS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "labels": MERGED_LABELS,
                    "prompt_templates": PROMPT_TEMPLATES,
                    "frames_per_video": FRAMES_PER_VIDEO,
                    "results": detailed_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved detailed predictions to: {PREDICTIONS_JSON_PATH}")


if __name__ == "__main__":
    main()
