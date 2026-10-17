import os

import numpy as np
import pytest

from app.detectors import DnnObjectDetector, StubDetector, YuNetFaceDetector

MODEL = "models/face_detection_yunet.onnx"
OBJ_MODEL = "models/object_detection_yolox.onnx"


def test_stub_detector_returns_centered_box():
    img = np.zeros((300, 600, 3), np.uint8)
    dets = StubDetector().detect(img)
    assert len(dets) == 1
    d = dets[0]
    assert d["label"] == "object"
    assert 0.0 <= d["confidence"] <= 1.0
    x, y, w, h = d["box"]
    # box lies inside the image
    assert 0 <= x and 0 <= y and x + w <= 600 and y + h <= 300


@pytest.mark.skipif(not os.path.exists(MODEL), reason="YuNet model not present")
def test_yunet_runs_on_blank_image():
    det = YuNetFaceDetector(MODEL)
    faces = det.detect(np.full((240, 320, 3), 200, np.uint8))
    assert isinstance(faces, list)  # blank image -> no faces, but must not crash


@pytest.mark.skipif(not os.path.exists(OBJ_MODEL), reason="YOLOX model not present")
def test_object_detector_runs():
    det = DnnObjectDetector(OBJ_MODEL)
    out = det.detect(np.full((480, 640, 3), 128, np.uint8))
    assert isinstance(out, list)  # blank frame -> no objects, must not crash
    for d in out:
        assert set(d) == {"label", "confidence", "box"}
