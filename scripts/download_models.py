"""Download the detection models into ./models.

- YuNet face model (~230 KB) — tiny, usually already committed.
- YOLOX object model (~35 MB) — needed for the default object detector.

Run:  python -m scripts.download_models
"""
from __future__ import annotations

import os
import urllib.request

MODELS = {
    "models/face_detection_yunet.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/object_detection_yolox.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "object_detection_yolox/object_detection_yolox_2022nov.onnx",
}


def main() -> None:
    os.makedirs("models", exist_ok=True)
    for path, url in MODELS.items():
        if os.path.exists(path):
            print(f"✓ {path} (already present)")
            continue
        print(f"downloading {path} ...")
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"✓ {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
