"""Image processing helpers: decode, dominant colours, annotate, encode."""
from __future__ import annotations

import base64
from typing import List

import numpy as np


def read_image(data: bytes) -> "np.ndarray":
    import cv2

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image (unsupported or corrupt file)")
    return img


def dominant_colors(image: "np.ndarray", k: int = 4) -> List[dict]:
    """Return the k most common colours via k-means, as hex + proportion."""
    import cv2

    pixels = image.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 50000:  # subsample large images for speed
        idx = np.random.default_rng(0).choice(len(pixels), 50000, replace=False)
        pixels = pixels[idx]
    k = max(1, min(k, len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    total = int(counts.sum()) or 1
    out = []
    for i in np.argsort(counts)[::-1]:
        b, g, r = (int(c) for c in centers[i])  # OpenCV is BGR
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}",
                    "proportion": round(float(counts[i] / total), 3)})
    return out


def annotate(image: "np.ndarray", detections: List[dict]) -> "np.ndarray":
    import cv2

    img = image.copy()
    for d in detections:
        x, y, w, h = d["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(img, f'{d["label"]} {d["confidence"]:.2f}', (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return img


def to_data_uri(image: "np.ndarray") -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
