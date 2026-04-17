"""
demo/realtime_webcam_fer.py

Purpose:
    Real-time webcam FER demo based on the RAVDESS-trained CNN-LSTM.
    Uses MediaPipe BlazeFace to detect the face, maintains a rolling
    buffer of the last SEQ_LEN face crops, and runs one clip-level
    forward pass per frame for 7-class emotion classification. An EMA
    smooths the predicted probability over time so the label does not
    flicker. Overlays bbox + label + confidence + top-left FPS +
    top-right 128x128 grayscale input preview on the video frame.

Pipeline:
    - Open the local webcam (CAMERA_ID = 0).
    - Detect faces with MediaPipe BlazeFace (blaze_face_short_range.tflite).
    - Keep the single face with the highest detection score.
    - Expand the bbox by 1.25x, crop, and convert to 128x128 grayscale.
    - Append the tensor to a deque of length BUFFER_LEN (raw buffer).
      Once the deque holds >= SEQ_LEN frames, uniformly sub-sample
      SEQ_LEN frames via torch.linspace and run CNNLSTM128 forward on
      the stacked clip [1, T, 1, H, W]. This mirrors the training-time
      sampling density in training/train_ravdess_cnn_lstm.py so the
      temporal stride roughly matches what the model is used to.
    - EMA-smooth the per-class probability and display the top-1 class.
    - Press q or Esc to quit.

Input:
    best_ravdess_cnn_lstm.pth              CNN-LSTM weights trained on RAVDESS frames.
    blaze_face_short_range.tflite          Pretrained MediaPipe face detector.
    Local webcam (CAMERA_ID = 0).

Output:
    Live OpenCV display window.

Usage (run from project root):
    python demo/realtime_webcam_fer.py
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. Config
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = str(PROJECT_ROOT / "weights" / "best_ravdess_cnn_lstm.pth")
MP_FACE_DETECTOR_MODEL = str(PROJECT_ROOT / "assets" / "blaze_face_short_range.tflite")   # MediaPipe face detector .task file
CAMERA_ID = 0

IMG_SIZE = 128
SEQ_LEN = 8           # frames fed into the CNN-LSTM per forward pass
BUFFER_LEN = 8        # raw face-crop buffer; equals SEQ_LEN = consecutive 8 frames.
                      # Tried 24 (linspace down to 8 to match training-stride ~3),
                      # but it pushed the model into a "disgust" attractor on
                      # webcam input. Keeping 8 = 8 consecutive frames as the
                      # stable default; bump up again only if combined with
                      # subject conditioning or a domain-adapted encoder.
FEAT_DIM = 256
HIDDEN_DIM = 256
BIDIRECTIONAL = True

EMA_ALPHA = 0.3       # new prediction weight for exponential moving average
FACE_MISS_RESET = 15  # consecutive face-miss frames to drop the buffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match training sorted() order (alphabetical).
CLASS_NAMES = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
WARMUP_COLOR = (0, 200, 255)
FACE_PREVIEW_SIZE = 160  # Top-right preview size for the model's input face image.


# =========================================================
# 2. CNN-LSTM model (mirrors training/train_ravdess_cnn_lstm.py)
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
# 3. Preprocessing
# =========================================================

# Match training-time transform: grayscale, 128x128, normalize to ~[-1, 1].
frame_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def load_cnn_lstm_model():
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


def preprocess_face_for_buffer(face_bgr):
    """
    face_bgr: cropped face in OpenCV BGR format.
    Returns:
        frame_tensor: [1, IMG_SIZE, IMG_SIZE] normalized tensor.
        gray_input_uint8: [IMG_SIZE, IMG_SIZE] uint8 grayscale preview
            (the un-normalized version of what the model sees).
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    x = frame_transform(pil_img)   # [1, H, W], normalized

    # For the on-screen preview, un-normalize back to [0, 255].
    preview = ((x.squeeze(0).cpu().numpy() * 0.5) + 0.5) * 255.0
    gray_input = preview.clip(0, 255).astype(np.uint8)

    return x, gray_input


@torch.no_grad()
def predict_clip_emotion(model, frame_buffer):
    """
    Uniformly sub-sample SEQ_LEN frames from `frame_buffer` via
    torch.linspace and run one clip-level forward pass. This matches
    training-time sampling in RAVDESSSequenceDataset._sample_frames.

    frame_buffer: a deque/list of >=SEQ_LEN normalized frame tensors,
                  each [1, H, W].
    Returns:
        probs: numpy array [num_classes] of softmax probabilities.
    """
    buffer_list = list(frame_buffer)
    n = len(buffer_list)

    if n >= SEQ_LEN:
        indices = torch.linspace(0, n - 1, steps=SEQ_LEN).long().tolist()
        selected = [buffer_list[i] for i in indices]
    else:
        # Defensive fallback (caller should guarantee n >= SEQ_LEN).
        selected = list(buffer_list)
        while len(selected) < SEQ_LEN:
            selected.append(buffer_list[-1])

    clip = torch.stack(selected, dim=0)              # [T, 1, H, W]
    clip = clip.unsqueeze(0).to(DEVICE)              # [1, T, 1, H, W]

    logits = model(clip)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy()


# =========================================================
# 4. Misc drawing helpers
# =========================================================

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def expand_box(x, y, bw, bh, img_w, img_h, scale=1.2):
    cx = x + bw / 2
    cy = y + bh / 2
    new_w = bw * scale
    new_h = bh * scale

    x1 = int(cx - new_w / 2)
    y1 = int(cy - new_h / 2)
    x2 = int(cx + new_w / 2)
    y2 = int(cy + new_h / 2)

    return clamp_box(x1, y1, x2, y2, img_w, img_h)


def create_face_detector():
    base_options = python.BaseOptions(model_asset_path=MP_FACE_DETECTOR_MODEL)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    detector = vision.FaceDetector.create_from_options(options)
    return detector


# =========================================================
# 5. Main realtime loop
# =========================================================

def main():
    print("Using device:", DEVICE)
    print("Loading CNN-LSTM model from:", MODEL_PATH)
    print("Loading MediaPipe model from:", MP_FACE_DETECTOR_MODEL)
    print(f"Raw buffer length: {BUFFER_LEN} frames | Clip length: {SEQ_LEN} frames")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"CNN-LSTM model not found: {MODEL_PATH}")
    if not Path(MP_FACE_DETECTOR_MODEL).exists():
        raise FileNotFoundError(f"MediaPipe model not found: {MP_FACE_DETECTOR_MODEL}")

    model = load_cnn_lstm_model()
    detector = create_face_detector()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Rolling buffer of the last BUFFER_LEN face tensors. Each inference
    # uniformly sub-samples SEQ_LEN frames from this buffer.
    frame_buffer = deque(maxlen=BUFFER_LEN)
    ema_probs = None
    miss_counter = 0

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            # Mirror for natural viewing.
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            img_h, img_w = frame.shape[:2]

            # MediaPipe expects RGB input.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            detection_result = detector.detect(mp_image)

            best_face_preview = None
            face_found = False
            current_bbox = None

            if detection_result.detections:
                # Keep only the face with the highest detection score.
                best_det = None
                best_score = -1.0

                for det in detection_result.detections:
                    score = det.categories[0].score if det.categories else 0.0
                    if score > best_score:
                        best_score = score
                        best_det = det

                bbox = best_det.bounding_box
                x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                x1, y1, x2, y2 = expand_box(x, y, bw, bh, img_w, img_h, scale=1.25)
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    face_tensor, gray_input = preprocess_face_for_buffer(face_crop)
                    frame_buffer.append(face_tensor)
                    face_found = True
                    miss_counter = 0
                    current_bbox = (x1, y1, x2, y2)

                    best_face_preview = cv2.resize(
                        gray_input,
                        (FACE_PREVIEW_SIZE, FACE_PREVIEW_SIZE),
                        interpolation=cv2.INTER_NEAREST,
                    )

            if not face_found:
                miss_counter += 1
                # If face has been missing for a while, drop the buffer so
                # stale frames do not poison the next prediction.
                if miss_counter >= FACE_MISS_RESET:
                    frame_buffer.clear()
                    ema_probs = None

            # Run clip-level inference once the raw buffer holds at least
            # SEQ_LEN frames; the sub-sampler will pick SEQ_LEN uniformly
            # spaced frames across whatever is currently buffered (up to
            # BUFFER_LEN).
            if len(frame_buffer) >= SEQ_LEN:
                new_probs = predict_clip_emotion(model, frame_buffer)

                if ema_probs is None:
                    ema_probs = new_probs
                else:
                    ema_probs = EMA_ALPHA * new_probs + (1.0 - EMA_ALPHA) * ema_probs

                pred_idx = int(np.argmax(ema_probs))
                pred_label = CLASS_NAMES[pred_idx]
                confidence = float(ema_probs[pred_idx])

                if current_bbox is not None:
                    x1, y1, x2, y2 = current_bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), BOX_COLOR, 2)
                    cv2.putText(
                        display,
                        f"{pred_label} ({confidence:.2f})",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        TEXT_COLOR,
                        2,
                        cv2.LINE_AA,
                    )
            else:
                # Still warming up: show how many frames we have.
                if current_bbox is not None:
                    x1, y1, x2, y2 = current_bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), WARMUP_COLOR, 2)

                cv2.putText(
                    display,
                    f"Warming up temporal buffer {len(frame_buffer)}/{SEQ_LEN} (raw {len(frame_buffer)}/{BUFFER_LEN})",
                    (10, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    WARMUP_COLOR,
                    2,
                    cv2.LINE_AA,
                )

            # Top-right preview of the 128x128 grayscale face the model sees.
            if best_face_preview is not None:
                preview_bgr = cv2.cvtColor(best_face_preview, cv2.COLOR_GRAY2BGR)
                ph, pw = preview_bgr.shape[:2]
                display[10:10 + ph, img_w - pw - 10:img_w - 10] = preview_bgr
                cv2.putText(
                    display,
                    f"FER input {IMG_SIZE}x{IMG_SIZE}",
                    (img_w - pw - 10, min(img_h - 10, 10 + ph + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # FPS overlay.
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Realtime FER (CNN-LSTM) with MediaPipe", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
