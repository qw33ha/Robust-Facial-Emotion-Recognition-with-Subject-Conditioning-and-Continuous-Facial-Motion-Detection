"""
preprocess/extract_ravdess_gray_frames.py

Purpose:
    Extract grayscale frames from RAVDESS videos, grouped by emotion into
    separate folders. The output is consumed by
    training/train_ravdess_cnn_actor_split.py.

Pipeline:
    - Walk ./RAVDESS and collect every .mp4 video.
    - Parse emotion and actor_id from the RAVDESS filename convention
      (calm (02) is merged into neutral (01), leaving 7 classes).
    - Skip the first 20 frames, then sample one frame every 5 frames.
    - Convert each sampled frame to grayscale and resize to 224x224.
    - Cap each (actor, emotion) pair at 200 frames to reduce class/actor
      imbalance.
    - Encode actor_id and emotion into the output filename so that
      downstream scripts can split by actor.

Input:
    ./RAVDESS/                      RAVDESS original video directory.

Output:
    ./output_gray_frames/<emotion>/<video>_actor_<id>_<emotion>_frame_<idx>.png

Usage (run from project root):
    python preprocess/extract_ravdess_gray_frames.py
"""

import os
import cv2
from pathlib import Path
from collections import defaultdict


RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_filename(filename: str):
    """
    Parse a RAVDESS filename.

    Example:
        02-01-06-01-02-01-12.mp4

    Returns:
        emotion_label: e.g. "fearful"
        actor_id: e.g. "12"
    """
    stem = Path(filename).stem
    parts = stem.split("-")

    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename: {filename}")

    emotion_id = parts[2]
    actor_id = parts[6]

    if emotion_id not in RAVDESS_EMOTION_MAP:
        raise ValueError(f"Unknown emotion id {emotion_id} in {filename}")

    emotion_label = RAVDESS_EMOTION_MAP[emotion_id] if emotion_id != "02" else RAVDESS_EMOTION_MAP["01"]
    return emotion_label, actor_id


def extract_limited_gray_frames(
    video_path: str,
    output_root: str,
    counters: dict,
    max_per_actor_emotion: int = 200,
    skip_first_n_frames: int = 100,
    sample_every_n_frames: int = 30,
    resize_to: tuple[int, int] | None = None,
) -> int:
    """
    Extract grayscale frames from a single video, capping the total number
    of frames saved for each (actor, emotion) pair at max_per_actor_emotion.
    """
    video_path = Path(video_path)
    video_name = video_path.stem

    emotion_label, actor_id = parse_ravdess_filename(video_path.name)
    key = (actor_id, emotion_label)

    # Skip the entire video if this (actor, emotion) has already hit the cap.
    if counters[key] >= max_per_actor_emotion:
        print(f"[SKIP] {video_name}: actor {actor_id}, emotion {emotion_label} already reached limit")
        return 0

    emotion_dir = Path(output_root) / emotion_label
    emotion_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stop early once the cap is reached mid-video.
        if counters[key] >= max_per_actor_emotion:
            break

        if frame_idx < skip_first_n_frames:
            frame_idx += 1
            continue

        if (frame_idx - skip_first_n_frames) % sample_every_n_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if resize_to is not None:
                gray = cv2.resize(gray, resize_to)

            # Encode actor_id and emotion in the output filename for easy
            # downstream splitting / debugging.
            out_name = (
                f"{video_name}_actor_{actor_id}_{emotion_label}"
                f"_frame_{frame_idx:05d}.png"
            )
            out_path = emotion_dir / out_name

            cv2.imwrite(str(out_path), gray)
            counters[key] += 1
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(
        f"[DONE] {video_name} -> {emotion_label}, actor {actor_id}: "
        f"saved {saved_count} frames "
        f"(total for this actor/emotion: {counters[key]}/{max_per_actor_emotion})"
    )
    return saved_count


def process_ravdess_videos(
    input_dir: str,
    output_root: str,
    max_per_actor_emotion: int = 200,
    skip_first_n_frames: int = 20,
    sample_every_n_frames: int = 5,
    resize_to: tuple[int, int] | None = None,
) -> None:
    """
    Batch-process RAVDESS videos, capping the number of grayscale frames
    saved for each (actor, emotion) pair.
    """
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    input_path = Path(input_dir)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    counters = defaultdict(int)
    total_videos = 0
    total_frames = 0

    # Sort for deterministic output.
    all_videos = sorted(
        [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in video_exts]
    )

    for file_path in all_videos:
        try:
            saved = extract_limited_gray_frames(
                video_path=str(file_path),
                output_root=str(output_path),
                counters=counters,
                max_per_actor_emotion=max_per_actor_emotion,
                skip_first_n_frames=skip_first_n_frames,
                sample_every_n_frames=sample_every_n_frames,
                resize_to=resize_to,
            )
            total_videos += 1
            total_frames += saved
        except Exception as e:
            print(f"[SKIP] {file_path.name}: {e}")

    print("\n===== Summary =====")
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames saved:     {total_frames}")

    print("\n===== Per actor x emotion counts =====")
    for (actor_id, emotion_label), count in sorted(counters.items()):
        print(f"Actor {actor_id}, {emotion_label}: {count}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_DIR = str(BASE_DIR / ".." / "RAVDESS")
    OUTPUT_DIR = str(BASE_DIR / ".." / "output_gray_frames")

    process_ravdess_videos(
        input_dir=INPUT_DIR,
        output_root=OUTPUT_DIR,
        max_per_actor_emotion=200,   # cap per (actor, emotion) pair
        skip_first_n_frames=20,      # drop the first 20 frames
        sample_every_n_frames=5,     # keep every 5th frame
        resize_to=(224, 224),        # uniform output size; set None to skip resize
    )
