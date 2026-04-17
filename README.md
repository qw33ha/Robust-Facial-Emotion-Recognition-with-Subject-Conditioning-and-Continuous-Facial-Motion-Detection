## Dataset
Download the three dataset and unzip them before running training; or use the trained weight to do evaluation directly:
https://drive.google.com/file/d/1gWsTg8qCEYShEPBffC_WgEGKbYqXiyrj/view?usp=sharing
https://drive.google.com/file/d/1q-DOtOk-pde9FYEimrj-xGu1YvaLKL90/view?usp=sharing
https://drive.google.com/file/d/1eU-yxf6vFCTvWwgSmvtGyhD9T0dWpIz4/view?usp=sharing

After unzipping, the project root should contain `FER2013/` and `RAVDESS/` directories. The pretrained checkpoints should be placed under `weights/`.

## Project structure

```
.
├── FER2013/                         # FER2013 static images (train/test)
├── RAVDESS/                         # RAVDESS original .mp4 videos
├── output_gray_frames/              # Grayscale frames extracted from RAVDESS
├── preprocess/
│   └── extract_ravdess_gray_frames.py
├── training/
│   ├── train_fer2013_shallow_cnn48.py
│   ├── train_ravdess_cnn_actor_split.py
│   ├── train_ravdess_cnn_lstm.py
│   └── train_ravdess_cnn_lstm_e2e.py
├── evaluation/
│   ├── eval_ravdess_cnn_on_frames.py
│   ├── eval_ravdess_cnn_lstm.py
│   ├── eval_ravdess_cnn_lstm_e2e.py
│   └── eval_ravdess_clip_zeroshot.py
├── weights/                         # Best checkpoints produced by training
├── outputs/                         # Confusion matrices, curves, JSON results
├── demo/
│   └── realtime_webcam_fer.py
└── assets/
```

## Preprocessing (RAVDESS only)

The three RAVDESS models (VGGStyleCNN128, CNN-LSTM warm-start, CNN-LSTM end-to-end) all consume the grayscale frames produced by the preprocessing script. **If you don't want to use the given frames,** run this **once** before training any RAVDESS model:

```bash
python preprocess/extract_ravdess_gray_frames.py
```

This walks `./RAVDESS`, samples one frame every 5 frames (after skipping the first 20), converts each sampled frame to 224x224 grayscale, caps each (actor, emotion) pair at 200 frames, merges `calm` into `neutral` (resulting in 7 classes), and writes the output to:

```
./output_gray_frames/<emotion>/<video>_actor_<id>_<emotion>_frame_<idx>.png
```

FER2013 is already in image form and does not require any preprocessing step.

## Models

| # | Model | Dataset | Input | Script |
|---|-------|---------|-------|--------|
| 1 | ShallowCNN48 | FER2013 | 48x48 grayscale image | `train_fer2013_shallow_cnn48.py` |
| 2 | VGGStyleCNN128 (frame-level) | RAVDESS | 128x128 grayscale image | `train_ravdess_cnn_actor_split.py` |
| 3 | CNN-LSTM (warm-start + frozen encoder) | RAVDESS | clip of 128x128 grayscale frames | `train_ravdess_cnn_lstm.py` |
| 4 | CNN-LSTM (end-to-end from scratch) | RAVDESS | clip of 128x128 grayscale frames | `train_ravdess_cnn_lstm_e2e.py` |

A zero-shot CLIP (ViT-B/32) baseline is also provided under `evaluation/` for reference; it requires no training.

All scripts below should be run from the **project root**.

---

### Model 1: ShallowCNN48 on FER2013 (static image, 7-class)

Architecture: 4 stages of `Conv + BN + ReLU + MaxPool` (channels 32 -> 64 -> 128 -> 256), then a 3-layer MLP head. Feature map is flattened directly, so the input resolution is hard-coded to 48x48. ~1.0M parameters.

Training recipe:
- 48x48 grayscale, random horizontal flip + random rotation (±10°).
- 10% of the training set is held out as validation.
- 20 epochs, Adam (lr=1e-3), CrossEntropy loss.
- Save the checkpoint with the best validation accuracy.
- After training, evaluate on the official test set and save the confusion matrix + training curve as PNGs.

Train:
```bash
python training/train_fer2013_shallow_cnn48.py
```

Outputs:
- `weights/best_fer2013_shallow_cnn48.pt`
- `outputs/fer_confusion_matrix.png`
- `outputs/fer_training_curve.png`

Test / evaluate: the training script already runs the final test-set evaluation at the end. 

---

### Model 2: VGGStyleCNN128 on RAVDESS frames (frame-level, actor-independent split)

Architecture: 4 double-conv stages (Conv -> ReLU -> [BN] -> Conv -> ReLU -> BN -> MaxPool -> Dropout2d(0.25)), channels 64 -> 128 -> 256 -> 512, followed by `AdaptiveAvgPool2d((2, 2))` and a 4-layer MLP head. ~5.9M parameters.

Training recipe:
- 128x128 grayscale frames from `./output_gray_frames/`.
- Actor-independent split: **actors 21/22/23/24** are the validation set; all other actors are training.
- 20 epochs, Adam (lr=1e-3, weight decay=1e-4), `ReduceLROnPlateau`.
- Every epoch prints val accuracy and UAR (from the confusion matrix).
- Save the checkpoint with the best validation accuracy.

Train:
```bash
# Make sure preprocess/extract_ravdess_gray_frames.py has been run first.
python training/train_ravdess_cnn_actor_split.py
```

Outputs:
- `weights/best_ravdess_actor_split_model.pth`

Test / evaluate (offline, standalone reproduction on the held-out actors 21-24):
```bash
python evaluation/eval_ravdess_cnn_on_frames.py
```
This loads `weights/best_ravdess_actor_split_model.pth`, runs per-image inference on frames belonging to actors 21-24, and prints overall accuracy, per-class accuracy, prediction distribution, and the confusion matrix (raw + row-normalized). It also saves `outputs/ravdess_cnn_confusion_matrix.png`.

---

### Model 3: CNN-LSTM on RAVDESS clips — warm-started + frozen encoder

Architecture:
- Encoder: the VGGStyleCNN128 feature extractor from Model 2 (+ adaptive pool + projection to `feat_dim`). Warm-started from `weights/best_ravdess_actor_split_model.pth` and **frozen**.
- Temporal aggregator: BiLSTM (num_layers=1, hidden=256) with attention pooling over the time axis.
- Head: 2-layer MLP with dropout.

Training recipe:
- Group frames into clips by the filename prefix `NN-NN-...-NN_actor_YY_<emotion>`.
- Same actor-independent split: actors 21/22/23/24 = val, rest = train.
- Uniformly sample `seq_len` frames per video via `torch.linspace`; if a video has fewer than `seq_len` frames, repeat the last one.
- 20 epochs, Adam (lr=1e-4, weight decay=1e-4), `ReduceLROnPlateau` on val accuracy.
- Only the temporal head (BiLSTM + attention + MLP) is learned; the CNN encoder stays frozen.
- On every new best val accuracy, save both the checkpoint and the confusion matrix of that validation pass.

Train:
```bash
# Requires both the grayscale frames and weights/best_ravdess_actor_split_model.pth.
python training/train_ravdess_cnn_lstm.py
```

Outputs:
- `weights/best_ravdess_cnn_lstm.pth`
- `outputs/ravdess_cnn_lstm_val_confusion.png` (best-epoch validation confusion matrix)

Test / evaluate (clip-level, actor-independent held-out split):
```bash
python evaluation/eval_ravdess_cnn_lstm.py
```
This loads `weights/best_ravdess_cnn_lstm.pth`, groups frames by video, for each video in actors 21-24 uniformly samples `SEQ_LEN` frames, runs one clip-level forward pass, and prints overall/per-class accuracy, prediction distribution, and the 7x7 confusion matrix. It also saves `outputs/ravdess_cnn_lstm_confusion_matrix.png`.

> `SEQ_LEN` at evaluation time must match the value used during training.

---

### Model 4: CNN-LSTM on RAVDESS clips — end-to-end from scratch

Same architecture and optimiser recipe as Model 3, with two differences:
- The VGGStyleCNN128 encoder is **randomly initialised and fully trainable** (no warm-start, no freezing).
- Every parameter (encoder + BiLSTM + attention pool + MLP head) is trained jointly.

This variant exists as the apples-to-apples comparison for the warm-started + frozen version: same data split, same architecture, same optimiser recipe; the only difference is whether the encoder is pretrained and frozen. Expected behaviour is comparable or lower val accuracy than Model 3, because the ~5M encoder parameters must be learned from only ~1440 videos.

Train:
```bash
python training/train_ravdess_cnn_lstm_e2e.py
```

Outputs:
- `weights/best_ravdess_cnn_lstm_e2e.pth`
- `outputs/ravdess_cnn_lstm_e2e_val_confusion.png` (best-epoch validation confusion matrix)

Test / evaluate (architecturally identical to Model 3's evaluation; only the checkpoint path and output names differ, so the numbers are directly comparable):
```bash
python evaluation/eval_ravdess_cnn_lstm_e2e.py
```
Outputs include `outputs/ravdess_cnn_lstm_e2e_confusion_matrix.png`.

---

### (Baseline) Zero-shot CLIP on RAVDESS

No training required. Uses HuggingFace `openai/clip-vit-base-patch32` directly on RAVDESS videos:
- Uniformly samples 5 frames per `.mp4`.
- Builds text embeddings for each class by averaging 3 prompt templates:
  - `"a facial expression showing {} emotion"`
  - `"a person with a {} facial expression"`
  - `"a face expressing {}"`
- Cosine similarity + softmax → per-frame class probabilities.
- Average over 5 frames → video-level prediction (neutral and calm merged into `neutral_or_calm`, 7 classes total).

Run:
```bash
python evaluation/eval_ravdess_clip_zeroshot.py
```

Outputs:
- `ravdess_confusion_matrix.png`
- `ravdess_clip_zero_shot_predictions.json`

---

## Real time demo
For a quick real-time sanity check using your webcam, run:
```bash
python demo/realtime_webcam_fer.py
```

---

## Recommended order of execution

1. Download and unzip FER2013, RAVDESS, and the weights archive.
2. `python preprocess/extract_ravdess_gray_frames.py` (only needed for RAVDESS models).
3. Train Model 1 (FER2013 / ShallowCNN48) — independent, can run in parallel with step 4.
4. Train Model 2 (RAVDESS / VGGStyleCNN128) — must finish before Model 3, because Model 3 warm-starts from this checkpoint.
5. Train Model 3 (CNN-LSTM warm-start + frozen) and Model 4 (CNN-LSTM end-to-end). These two are independent of each other.
6. Run the matching `evaluation/eval_*.py` script for any trained model, or the zero-shot CLIP baseline, to reproduce the reported numbers and confusion matrices.

**If you only want to reproduce evaluation numbers, skip steps 3-5 and use the provided `weights/*.pt(h)` checkpoints with the `evaluation/` scripts directly.**
