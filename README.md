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
│   ├── train_ravdess_cnn_lstm_e2e.py
│   ├── train_ravdess_cnn_lstm_residue.py
│   └── train_ravdess_cnn_lstm_residue_concat.py
├── evaluation/
│   ├── eval_ravdess_cnn_on_frames.py
│   ├── eval_ravdess_cnn_lstm.py
│   ├── eval_ravdess_cnn_lstm_e2e.py
│   ├── eval_ravdess_cnn_lstm_residue.py
│   ├── eval_ravdess_cnn_lstm_residue_concat.py
│   └── eval_ravdess_clip_zeroshot.py
├── weights/                         # Best checkpoints produced by training
├── outputs/                         # Confusion matrices, curves, JSON results
├── demo/
│   ├── realtime_webcam_fer.py              # Model 3 (no residue)
│   ├── realtime_webcam_model5.py           # Model 5 (pure residue, with enrollment)
│   └── realtime_webcam_model5b.py          # Model 5b (residue-concat, with enrollment)
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
| 5 | CNN-LSTM + subject-residue (pure, `f_t − f_neutral`) | RAVDESS | clip of 128x128 grayscale frames + per-actor neutral template | `train_ravdess_cnn_lstm_residue.py` |
| 5b | CNN-LSTM + subject-residue (concat, `[f_t, f_t − f_neutral]`) | RAVDESS | clip of 128x128 grayscale frames + per-actor neutral template | `train_ravdess_cnn_lstm_residue_concat.py` |

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

### Model 5: CNN-LSTM with subject-residue conditioning

Same architecture as Model 3 (warm-started + frozen VGGStyleCNN128 encoder, BiLSTM, attention pool, MLP head, identical optimiser and epochs), with one extra step inserted between the encoder and the BiLSTM:

```
f_t      = Encoder(x_t)                # per-frame feature, [B, T, D]
f_actor  = mean( Encoder(x_neutral_k) ) # per-actor neutral template, [B, D]
residue  = f_t − f_actor               # subject-normalized feature, [B, T, D]
logits   = Classifier(AttnPool(BiLSTM(residue)))
```

The intent is to subtract the actor's identity / face shape out of the per-frame embedding so the temporal head only sees the **expression-induced** component. Output dimensionality is unchanged (still `feat_dim`), so all of Model 3's hyperparameters carry over without modification.

Subject template details:
- **Source**: the `neutral` emotion folder produced by `preprocess/extract_ravdess_gray_frames.py` (which already merges RAVDESS `calm` into `neutral`).
- **Encoding**: every neutral frame is passed through the (frozen) warm-started encoder once, before training starts. This bank lives in CPU memory; for 24 actors capped at 200 frames each, it's ~5K×256 floats.
- **Training-time sampling**: for each clip in the batch, randomly sample `K=8` rows from that actor's neutral bank and take their mean → `f_actor`. The randomness acts as a mild data augmentation on the template.
- **Validation/test-time template**: deterministic mean of **all** neutral frames of that actor — no random sampling, fully reproducible.
- **About held-out actors (21-24)**: their neutral frames are used to build their templates, but their non-neutral expression labels never enter training. This matches standard identity-aware FER protocol: the model is allowed to "see the person's resting face" (enrollment) but not their emotional expressions.

Training recipe (identical to Model 3 except for the residue step and the K=8 neutral sampler):
- Actor-independent split: actors 21/22/23/24 = val, rest = train.
- Uniformly sample `seq_len=8` frames per clip via `torch.linspace`; pad with the last frame if needed.
- 20 epochs, Adam (lr=1e-4, weight decay=1e-4), `ReduceLROnPlateau` on val accuracy.
- Encoder frozen (BN running stats pinned to the warm-start values); only BiLSTM + attention + MLP are trained.
- Save best-by-val-acc checkpoint and the corresponding confusion matrix PNG.

Train:
```bash
# Requires both the grayscale frames and weights/best_ravdess_actor_split_model.pth.
python training/train_ravdess_cnn_lstm_residue.py
```

Outputs:
- `weights/best_ravdess_cnn_lstm_residue.pth`
- `outputs/ravdess_cnn_lstm_residue_val_confusion.png` (best-epoch validation confusion matrix)

Test / evaluate (clip-level, actor-independent held-out split; the eval script rebuilds the per-actor neutral template from the checkpoint encoder, so the number is reproducible from `weights/` + `output_gray_frames/` alone):
```bash
python evaluation/eval_ravdess_cnn_lstm_residue.py
```
This prints overall / per-class / **per-actor** accuracy (the per-actor breakdown is the most informative when comparing against Model 3 — a residue trick that "averages well" but breaks one specific actor is a red flag), plus the prediction distribution and the 7×7 confusion matrix. Saves `outputs/ravdess_cnn_lstm_residue_confusion_matrix.png`.

> Comparison protocol: keep `seq_len`, `feat_dim`, `hidden_dim`, optimiser, epochs, and the actor split identical between Model 3 and Model 5. The val-accuracy / per-actor-accuracy delta then cleanly attributes to subject conditioning.

---

### Model 5b: CNN-LSTM with subject-residue — **concat** variant

Model 5 replaces `f_t` with the pure residue `f_t − f_neutral`. In practice this biases the classifier toward **high-magnitude** expressions (in our runs, angry and happy recalls saturate around 97%) at the cost of **low-magnitude** ones (disgust / surprised collapse). The geometric cause is simple: pure subtraction discards the absolute magnitude of `f_t`, which low-intensity expressions rely on for discriminability.

Model 5b keeps both signals by **concatenating** the raw feature and the residue before the BiLSTM:

```
f_t       = Encoder(x_t)                    # [B, T, D]
f_actor   = mean( Encoder(x_neutral_k) )    # [B, D]
fused_t   = [ f_t ,  f_t − f_actor ]        # [B, T, 2D]
logits    = Classifier(AttnPool(BiLSTM(fused_t)))
```

The BiLSTM can then decide on its own when to rely on raw feature strength vs when to rely on the subject-normalized direction. The only structural change vs Model 5 is that `LSTM.input_size` doubles from `feat_dim` (256) to `2 * feat_dim` (512); hidden size, attention pool, MLP head, optimiser, and all data / split / enrollment conventions are unchanged.

Same training recipe as Model 5:
- Warm-started + frozen VGGStyleCNN128 encoder, BN running stats pinned.
- Actor-independent split (actors 21-24 = val), `seq_len = 8`.
- 20 epochs, Adam (lr=1e-4, weight decay=1e-4), `ReduceLROnPlateau` on val accuracy.
- Training-time `K = 8` random neutral sampling per clip; validation-time deterministic mean over all neutral frames of the actor.

Train:
```bash
# Requires both the grayscale frames and weights/best_ravdess_actor_split_model.pth.
python training/train_ravdess_cnn_lstm_residue_concat.py
```

Outputs:
- `weights/best_ravdess_cnn_lstm_residue_concat.pth`
- `outputs/ravdess_cnn_lstm_residue_concat_val_confusion.png` (best-epoch validation confusion matrix)

Test / evaluate (clip-level, actor-independent held-out split; the eval script rebuilds the per-actor neutral template from the checkpoint encoder):
```bash
python evaluation/eval_ravdess_cnn_lstm_residue_concat.py
```
Prints overall / per-class / per-actor accuracy, the prediction distribution, and the 7×7 confusion matrix (counts + row-normalized %). Saves `outputs/ravdess_cnn_lstm_residue_concat_confusion_matrix.png`.

> Read this variant as a knob between Model 3 (no residue, raw features only) and Model 5 (residue only). If Model 5b beats both on val UAR **and** recovers the low-magnitude classes that Model 5 collapsed, the concat hypothesis is validated: the classifier needs both identity-normalized direction and absolute feature strength.

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

Three webcam demos are provided, one per architecture, so you can watch the effect of subject-residue conditioning (and of the pure-vs-concat residue choice) live against the non-conditioned baseline.

**Demo A — Model 3 (no residue):**
```bash
python demo/realtime_webcam_fer.py
```
Loads `weights/best_ravdess_cnn_lstm.pth`, runs MediaPipe BlazeFace for face detection, maintains a rolling 8-frame buffer, feeds it to the warm-started + frozen CNN-LSTM, and overlays an EMA-smoothed softmax bar chart. Good for a quick sanity check on the non-residue baseline.

**Demo B — Model 5 (subject-residue, with live enrollment):**
```bash
python demo/realtime_webcam_model5.py
```
Loads `weights/best_ravdess_cnn_lstm_residue.pth`. Because the residue model needs a per-subject neutral template to subtract, the demo exposes three template modes:

- **GENERIC** (default at startup): the template is the population mean computed by encoding every neutral frame in `output_gray_frames/neutral/` through the checkpoint encoder once, capped at `GENERIC_MAX_FRAMES_PER_ACTOR` frames per actor. This is the "unknown user" fallback — the 24 RAVDESS actors are averaged into a single generic neutral.
- **PERSONAL**: press **N** to start a ~3-second enrollment while holding a resting / neutral face; the demo encodes ~30 frames through the frozen encoder and averages them into a user-specific template. Progress bar + banner make the enrollment state obvious.
- Press **R** to reset back to the GENERIC template.

Key bindings during the demo:
- **N** — start (or restart) personal enrollment
- **R** — reset to GENERIC template
- **[** — shrink crop (both dims proportionally)
- **]** — grow crop (both dims proportionally)
- **\\** — reset crop scales to defaults
- **Q** — quit

This matches the residue model's training protocol: the val-time template is a deterministic mean of neutral frames for the held-out actors, so a live personal enrollment is the closest possible analogue for a new user.

**Tuning the crop (important).** `preprocess/extract_ravdess_gray_frames.py` does NOT face-crop — it just grayscales and resizes the full RAVDESS video frame (head + shoulders + studio background). A tight BlazeFace-only crop at demo time is therefore a large distribution shift for the encoder, and on Model 5 it tends to collapse the classifier into a 2-class attractor (e.g. happy ↔ disgust). The demo consequently ships with an asymmetric, downward-biased expansion of the BlazeFace bbox — roughly `W × 2.2`, `H × 2.8`, with the crop center shifted ~30% of the face height downward — so the 128×128 input contains the whole head + neck + shoulders + some background, approximating RAVDESS framing.

If the predictions still seem stuck, use `[` / `]` to adjust the crop live until the top-right 128×128 input preview looks like a RAVDESS-style head-and-shoulders shot (not a tight face close-up). The rolling frame buffer and EMA state are cleared on every scale change, so a fresh prediction starts after a short warm-up. Tune with `[` / `]` **before** enrolling with `N`, otherwise your personal neutral template will be computed under a crop different from the one you end up using for inference.

> If `weights/best_ravdess_cnn_lstm_residue.pth` or the `output_gray_frames/neutral/` folder are missing, Demo B will fail to start. Train Model 5 (or copy its provided checkpoint into `weights/`) and make sure the grayscale preprocessing has been run first.

**Demo C — Model 5b (subject-residue concat, with live enrollment) — recommended:**
```bash
python demo/realtime_webcam_model5b.py
```
Loads `weights/best_ravdess_cnn_lstm_residue_concat.pth`. Architecturally identical to Demo B except the BiLSTM input is the concatenation `[f_t, f_t − f_neutral]` (size `2 * feat_dim = 512`) instead of the pure residue (size `feat_dim = 256`). All template logic, key bindings, crop tuning, and enrollment behaviour are the same as Demo B:

- **GENERIC** (default at startup): population mean over every neutral frame in `output_gray_frames/neutral/`, capped at `GENERIC_MAX_FRAMES_PER_ACTOR` per actor.
- **PERSONAL**: press **N** to start ~3-second enrollment with a resting face; ~30 frames are encoded and averaged into a user-specific template.
- Press **R** to revert to the GENERIC template.

Key bindings during the demo (identical to Demo B):
- **N** — start (or restart) personal enrollment
- **R** — reset to GENERIC template
- **[** — shrink crop (both dims proportionally)
- **]** — grow crop (both dims proportionally)
- **\\** — reset crop scales to defaults
- **Q** — quit

Why this is the recommended demo: in our offline evaluation, the pure-residue Model 5 produced very high recall on high-magnitude expressions (angry / happy ≈ 97%) but collapsed on low-magnitude ones (disgust ≈ 28%). Model 5b's concat fusion narrows that spread substantially (disgust recovers to ~66%) at a comparable overall val accuracy. On webcam input — which is itself a domain shift away from the full-frame RAVDESS training distribution — preserving the absolute feature magnitude alongside the subject-normalized direction makes the live predictions noticeably more stable, with fewer 2-class attractor lock-ups.

The same crop-tuning advice from Demo B applies verbatim (the encoder was trained on full RAVDESS video frames, not face crops); use `[` / `]` to make the 128×128 input preview look like a head-and-shoulders shot before enrolling with `N`.

> If `weights/best_ravdess_cnn_lstm_residue_concat.pth` or the `output_gray_frames/neutral/` folder are missing, Demo C will fail to start. Train Model 5b (or copy its provided checkpoint into `weights/`) and make sure the grayscale preprocessing has been run first.

---

## Recommended order of execution

1. Download and unzip FER2013, RAVDESS, and the weights archive.
2. `python preprocess/extract_ravdess_gray_frames.py` (only needed for RAVDESS models).
3. Train Model 1 (FER2013 / ShallowCNN48) — independent, can run in parallel with step 4.
4. Train Model 2 (RAVDESS / VGGStyleCNN128) — must finish before Models 3 and 5, because both warm-start their encoder from this checkpoint.
5. Train Model 3 (CNN-LSTM warm-start + frozen), Model 4 (CNN-LSTM end-to-end), Model 5 (CNN-LSTM + subject-residue, pure), and Model 5b (CNN-LSTM + subject-residue, concat). Models 3, 5, and 5b all need the Model 2 checkpoint; Model 4 does not. All four are independent of each other and can run sequentially or in parallel.
6. Run the matching `evaluation/eval_*.py` script for any trained model, or the zero-shot CLIP baseline, to reproduce the reported numbers and confusion matrices.
7. (Optional) Run `demo/realtime_webcam_fer.py` (Model 3), `demo/realtime_webcam_model5.py` (Model 5, pure residue, with personal enrollment), or `demo/realtime_webcam_model5b.py` (Model 5b, residue-concat, with personal enrollment — recommended) for a live webcam sanity check.

**If you only want to reproduce evaluation numbers, skip steps 3-5 and use the provid