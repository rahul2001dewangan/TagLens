"""Pluggable object/face detectors.

The rest of the app depends only on `detect(image) -> list[dict]`, where each
dict is {"label": str, "confidence": float, "box": [x, y, w, h]}.

- YuNetFaceDetector: real face detection via OpenCV's modern FaceDetectorYN
  (works on OpenCV 5.x; needs a small ~230 KB ONNX model in models/).
- StubDetector: deterministic, no model needed (tests / offline demo).
- YOLODetector: richer object detection if `ultralytics` is installed.

get_detector() never raises: if the requested detector can't be built
(missing model, unsupported build), it logs a warning and falls back to the stub.
"""
from __future__ import annotations

import logging
import os
from typing import List, Protocol

import numpy as np

logger = logging.getLogger("taglens")


class Detector(Protocol):
    name: str

    def detect(self, image: "np.ndarray") -> List[dict]: ...


class StubDetector:
    """Deterministic detector for tests and a zero-setup demo."""

    name = "stub"

    def detect(self, image: "np.ndarray") -> List[dict]:
        h, w = image.shape[:2]
        bw, bh = max(1, w // 3), max(1, h // 3)
        return [{
            "label": "object",
            "confidence": 0.99,
            "box": [w // 2 - bw // 2, h // 2 - bh // 2, bw, bh],
        }]


class YuNetFaceDetector:
    """Face detection using OpenCV's FaceDetectorYN (YuNet)."""

    name = "yunet-face"

    def __init__(self, model_path: str, min_confidence: float = 0.6):
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("this OpenCV build has no FaceDetectorYN")
        if not os.path.exists(model_path):
            raise RuntimeError(f"YuNet model not found at {model_path}")
        self.cv2 = cv2
        self.min_conf = max(0.0, float(min_confidence))
        score_threshold = max(0.5, self.min_conf)
        self.fd = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold, 0.3, 5000
        )

    # YuNet detects poorly on very large frames; downscale wide images first,
    # run detection, then map the boxes back to the original coordinates.
    TARGET_WIDTH = 640

    def detect(self, image: "np.ndarray") -> List[dict]:
        h, w = image.shape[:2]
        if w > self.TARGET_WIDTH:
            scale = self.TARGET_WIDTH / w
            proc = self.cv2.resize(image, (self.TARGET_WIDTH, max(1, int(h * scale))))
        else:
            scale = 1.0
            proc = image
        ph, pw = proc.shape[:2]
        self.fd.setInputSize((pw, ph))
        _, faces = self.fd.detect(proc)
        out = []
        if faces is None:
            return out
        inv = 1.0 / scale
        for f in faces:
            x, y, bw, bh = f[:4]
            box = [max(0, int(x * inv)), max(0, int(y * inv)),
                   int(bw * inv), int(bh * inv)]
            out.append({"label": "face", "confidence": round(float(f[-1]), 2), "box": box})
        return out


class YOLODetector:
    name = "yolo"

    def __init__(self, model_path: str, min_confidence: float = 0.3):
        from ultralytics import YOLO  # optional dependency

        self.model = YOLO(model_path)
        self.min_confidence = min_confidence

    def detect(self, image: "np.ndarray") -> List[dict]:
        result = self.model(image, verbose=False)[0]
        out = []
        for b in result.boxes:
            conf = float(b.conf[0])
            if conf < self.min_confidence:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            out.append({
                "label": self.model.names[int(b.cls[0])],
                "confidence": round(conf, 2),
                "box": [x1, y1, x2 - x1, y2 - y1],
            })
        return out


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class DnnObjectDetector:
    """Object detection (80 COCO classes) via YOLOX + OpenCV DNN — no PyTorch.

    Works on Python/OpenCV combos where ultralytics/torch have no wheels.
    """

    name = "yolox"
    INPUT = 640

    def __init__(self, model_path: str, min_confidence: float = 0.5, nms: float = 0.5):
        if not os.path.exists(model_path):
            raise RuntimeError(f"object model not found at {model_path}")
        import cv2  # noqa: F401 (ensure OpenCV present)

        from ._yolox import YoloX

        self.cv2 = cv2
        self.model = YoloX(model_path, confThreshold=min_confidence,
                           nmsThreshold=nms, objThreshold=min_confidence)

    def _letterbox(self, image):
        h, w = image.shape[:2]
        r = min(self.INPUT / h, self.INPUT / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((self.INPUT, self.INPUT, 3), 114, np.uint8)
        canvas[:nh, :nw] = self.cv2.resize(image, (nw, nh))
        return canvas, r

    def detect(self, image: "np.ndarray") -> List[dict]:
        blob, r = self._letterbox(image)
        dets = self.model.infer(blob.astype(np.float32))
        out = []
        if dets is None or len(dets) == 0:
            return out
        inv = 1.0 / r
        for d in dets:
            x, y, w, h, score, cls = d
            cls = int(cls)
            label = COCO_CLASSES[cls] if 0 <= cls < len(COCO_CLASSES) else str(cls)
            out.append({
                "label": label,
                "confidence": round(float(score), 2),
                "box": [max(0, int(x * inv)), max(0, int(y * inv)),
                        int(w * inv), int(h * inv)],
            })
        return out


def get_detector(settings) -> Detector:
    """Build the configured detector, falling back to faces then stub on failure."""

    def _build(name: str) -> Detector:
        if name == "stub":
            return StubDetector()
        if name in ("objects", "yolox", "dnn"):
            return DnnObjectDetector(settings.object_model, settings.min_confidence)
        if name == "yolo":
            return YOLODetector(settings.yolo_model, settings.min_confidence)
        return YuNetFaceDetector(settings.yunet_model, settings.min_confidence)  # yunet/haar

    # Try the requested detector, then face detection, then the stub.
    for choice in (settings.detector, "yunet", "stub"):
        try:
            det = _build(choice)
            if choice != settings.detector:
                logger.warning("detector '%s' unavailable; using '%s' instead.",
                               settings.detector, det.name)
            return det
        except Exception as exc:
            logger.warning("detector '%s' failed to load: %s", choice, exc)
    return StubDetector()
