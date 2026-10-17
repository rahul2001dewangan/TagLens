"""Generate small sample media so the app/pipeline runs out of the box.

Creates:
  data/sample_image.jpg  - simple shapes (good for the stub detector / colour demo)
  data/sample_video.avi  - a few seconds of a moving shape (MJPG, widely supported)

Run:  python -m scripts.make_samples
For real face detection (DETECTOR=haar), upload your own photo in the UI.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

OUT_DIR = "data"


def make_image(path: str) -> None:
    img = np.full((360, 480, 3), 240, np.uint8)
    cv2.rectangle(img, (40, 40), (200, 200), (180, 90, 40), -1)
    cv2.circle(img, (340, 160), 90, (40, 160, 220), -1)
    cv2.putText(img, "TagLens sample", (40, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)


def make_video(path: str, seconds: int = 3, fps: int = 20) -> None:
    w, h = 480, 360
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        print("warning: could not open video writer; skipping sample video")
        return
    for i in range(seconds * fps):
        frame = np.full((h, w, 3), 240, np.uint8)
        x = int((i / (seconds * fps)) * (w - 80))
        cv2.rectangle(frame, (x, 140), (x + 80, 220), (40, 160, 220), -1)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_image(os.path.join(OUT_DIR, "sample_image.jpg"))
    make_video(os.path.join(OUT_DIR, "sample_video.avi"))
    print(f"Samples written to ./{OUT_DIR}/")
