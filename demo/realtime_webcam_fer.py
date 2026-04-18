"""
demo/realtime_webcam_fer.py

Purpose:
    Real-time webcam FER demo based on the FER2013-trained ShallowCNN48
    (Model 1 in the report). Uses MediaPipe BlazeFace to detect the face
    and runs one single-frame forward pass per webcam tick for 7-class
    emotion classification. An EMA smooths the predicted probability
    over time so the label does not flicker. Overlays bbox + label +
    confidence + top-left FPS + top-right 48x48 grayscale input preview
    on the video frame.

Pipeline:
    - Open the local webcam (CAMERA_ID = 0).
    - Detect faces with MediaPipe BlazeFace (blaze_face_short_range.tflite).
    - Keep the single face with the highest detection score.
    - Expand the bbox by 1.25x, crop, and convert to 48x48 grayscale
      following the FER2013 test-time transform (Grayscale -> Resize ->
      ToTensor, no normalization, to match training/train_fer2013_shallow_cnn48.py).
    - Forward through ShallowCNN48, take softmax, EMA-smooth the
      per-class probability and display the top-1 class.
    - Press q or Esc to quit.

Input:
    best_fer2013_shallow_cnn48.pt          ShallowCNN48 weights trained on FER2013.
    blaze_face_short_range.tflite          Pretrained MediaPipe face detector.
    Local webcam (CAMERA_ID = 0).

Output:
    Live OpenCV display window.

Usage (run from project root):
    python demo/realtime_webcam_fer.py
"""

import time
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
MODEL_PATH = str(PROJECT_ROOT / "weights" / "best_fer2013_shallow_cnn48.pt")
MP_FACE_DETECTOR_MODEL = str(PROJECT_ROOT / "assets" / "blaze_face_short_range.tflite")   # MediaPipe face detector .task file
CAMERA_ID = 0

IMG_SIZE = 48         # ShallowCNN48 expects 48x48 grayscale input.
NUM_CLASSES = 7

EMA_ALPHA = 0.3       # new prediction weight for exponential moving average
FACE_MISS_RESET = 15  # consecutive face-miss frames to reset the EMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match training sorted() order from datasets.ImageFolder (alphabetical).
# FER2013 uses "fear"/"surprise" (not "fearful"/"surprised" like RAVDESS).
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
FACE_PREVIEW_SIZE = 160  # Top-right preview size for the model's input face image.


# =========================================================
# 2. ShallowCNN48 model (mirrors training/train_fer2013_shallow_cnn48.py)
# =========================================================

class ShallowCNN48(nn.Module):
    """
    Shallow 4-stage CNN for 48x48 grayscale FER2013 input.

    - Single Conv per stage (Conv + BN + ReLU + MaxPool).
    - Channels: 32 -> 64 -> 128 -> 256.
    - No feature-map dropout.
    - Feature map is flattened directly (no adaptive pool), so the input
      resolution is hard-coded to 48x48.
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
# 3. Preprocessing
# =========================================================

# Match training-time test transform exactly: grayscale, 48x48, ToTensor.
# NOTE: training script does NOT normalize beyond ToTensor() (which maps
# uint8 -> [0, 1]). Any extra Normalize() here would silently shift the
# input distribution and confuse the BN statistics.
frame_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_shallow_cnn_model():
    model = ShallowCNN48(num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_face(face_bgr):
    """
    face_bgr: cropped face in OpenCV BGR format.
    Returns:
        frame_tensor: [1, IMG_SIZE, IMG_SIZE] tensor in [0, 1].
        gray_input_uint8: [IMG_SIZE, IMG_SIZE] uint8 grayscale preview
            (what the model actually sees).
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    x = frame_transform(pil_img)   # [1, H, W], in [0, 1]

    # For the on-screen preview, scale back to [0, 255].
    preview = x.squeeze(0).cpu().numpy() * 255.0
    gray_input = preview.clip(0, 255).astype(np.uint8)

    return x, gray_input


@torch.no_grad()
def predict_frame_emotion(model, frame_tensor):
    """
    Run a single 48x48 grayscale face tensor through ShallowCNN48.

    frame_tensor: [1, H, W] tensor in [0, 1].
    Returns:
        probs: numpy array [num_classes] of softmax probabilities.
    """
    x = frame_tensor.unsqueeze(0).to(DEVICE)  # [1, 1, H, W]
    logits = model(x)
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
    print("Loading ShallowCNN48 model from:", MODEL_PATH)
    print("Loading MediaPipe model from:", MP_FACE_DETECTOR_MODEL)
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE} grayscale | classes: {CLASS_NAMES}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"ShallowCNN48 model not found: {MODEL_PATH}")
    if not Path(MP_FACE_DETECTOR_MODEL).exists():
        raise FileNotFoundError(f"MediaPipe model not found: {MP_FACE_DETECTOR_MODEL}")

    model = load_shallow_cnn_model()
    detector = create_face_detector()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

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
            new_probs = None

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
                    face_tensor, gray_input = preprocess_face(face_crop)
                    new_probs = predict_frame_emotion(model, face_tensor)
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
                # If face has been missing for a while, drop the EMA so
                # stale probabilities do not linger.
                if miss_counter >= FACE_MISS_RESET:
                    ema_probs = None

            if new_probs is not None:
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

            # Top-right preview of the 48x48 grayscale face the model sees.
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

            cv2.imshow("Realtime FER (ShallowCNN48 / FER2013) with MediaPipe", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
