import os
import tempfile

os.environ["DETECTOR"] = "stub"  # offline, deterministic — set before importing app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _png_bytes(w=320, h=240):
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.rectangle(img, (20, 20), (120, 120), (0, 0, 255), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_home_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "TagLens" in r.text


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["detector"] == "stub"


def test_analyze_image():
    r = client.post("/analyze/image", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 320 and body["height"] == 240
    assert "object" in body["tags"]            # stub always returns one 'object'
    assert body["tag_counts"]["object"] == 1
    assert len(body["dominant_colors"]) >= 1
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")


def test_analyze_image_rejects_garbage():
    r = client.post("/analyze/image", files={"file": ("x.png", b"not-an-image", "image/png")})
    assert r.status_code == 400


def test_analyze_video_best_effort():
    # Generating a video needs an available codec; skip gracefully if unavailable.
    path = os.path.join(tempfile.mkdtemp(), "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    if not writer.isOpened():
        pytest.skip("no video writer/codec available in this environment")
    for _ in range(20):
        writer.write(np.full((120, 160, 3), 128, np.uint8))
    writer.release()
    with open(path, "rb") as f:
        r = client.post("/analyze/video", files={"file": ("clip.avi", f.read(), "video/avi")})
    assert r.status_code == 200
    body = r.json()
    assert body["frames_analyzed"] >= 1
    assert body["fps"] > 0
