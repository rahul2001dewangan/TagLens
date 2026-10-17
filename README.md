# TagLens — Auto Media Tagger

TagLens analyzes the **content of an image or a short video** and returns
auto-generated **tags** (objects / faces), basic **image analytics** (dominant
colours), and, for video, **per-tag screen-time** and a timeline. It's a small
version of the "content indexing / auto-tagging" workflow used in media
platforms.

- **Image** → detections (boxes + labels + confidence), tags, dominant colours, an annotated preview.
- **Video** (short clip) → frames are sampled and detected; you get screen-time per tag, detection counts, and a timeline.

## Why it's interesting
- **Real object detection with no PyTorch** — the default detector runs **YOLOX
  (80 COCO classes)** through OpenCV's **DNN** module, so it works even on Python
  versions where ultralytics/torch have no wheels.
- **Pluggable detector** (one env var, no code change):
  - `objects` — 80 COCO classes via YOLOX + OpenCV DNN (**default**)
  - `yunet` — face detection via OpenCV's `FaceDetectorYN` (tiny model, bundled)
  - `stub` — deterministic, offline (used by tests/CI)
  - `yolo` — ultralytics YOLO (needs PyTorch)
- **Fail-safe** — if a detector's model is missing it falls back (objects → faces →
  stub) with a logged warning instead of crashing.
- **Image processing + ML + analytics** in one small, readable pipeline.

## Architecture
```
upload ──► decode image / sample video frames   (app/imaging.py, app/video.py)
       ──► detector.detect(frame)               (app/detectors.py: haar | stub | yolo)
       ──► aggregate: tags, counts, screen-time  (app/analytics.py)
       ──► dominant colours + annotated preview  (app/imaging.py)
       ──► JSON + web UI                         (app/main.py, app/web/index.html)
```

## Quickstart
```bash
pip install -r requirements-dev.txt
python -m scripts.download_models      # fetches the YOLOX (~35MB) + YuNet models
uvicorn app.main:app --reload
```
Open **http://localhost:8000/** (web UI) or **/docs** (API).
- Upload an **image** → it detects objects/people, draws boxes, lists tags + dominant colours.
- Switch to **Video** and upload a short clip → per-tag screen-time + timeline.

**What it detects:** by default, **~80 everyday object classes** (person, car,
bicycle, dog, cat, bottle, cup, chair, laptop, cell phone, …) via YOLOX. Set
`DETECTOR=yunet` for face detection instead (its model is bundled, no download).

Optional sample media:
```bash
python -m scripts.make_samples      # writes data/sample_image.jpg, data/sample_video.avi
```

Richer object tags (optional):
```bash
pip install ultralytics
DETECTOR=yolo uvicorn app.main:app --reload   # downloads a small YOLO model on first run
```

## API
| Method | Path             | Purpose                                            |
|--------|------------------|----------------------------------------------------|
| GET    | `/`              | web UI                                             |
| GET    | `/health`        | status + active detector                           |
| POST   | `/analyze/image` | multipart `file` → detections, tags, colours, preview |
| POST   | `/analyze/video` | multipart `file` → screen-time, counts, timeline   |

## Tests
```bash
make test        # DETECTOR=stub pytest -q  (offline; no model needed)
```
Tests cover the detector contract, analytics aggregation, and the API end-to-end.
CI (`.github/workflows/ci.yml`) runs them with the stub detector.

## Project layout
```
app/
  config.py      settings (env / .env)
  detectors.py   objects (YOLOX) | yunet (faces) | stub | yolo — fail-safe chain
  _yolox.py      YOLOX ONNX inference via OpenCV DNN (vendored, Apache-2.0)
  imaging.py     decode, dominant colours, annotate, encode
  video.py       frame sampling for short clips
  analytics.py   tags, counts, screen-time aggregation
  main.py        FastAPI app + web UI
  web/index.html single-page UI
scripts/
  download_models.py  fetch YOLOX + YuNet models
  make_samples.py     generate sample image/video
tests/                detectors, analytics, API
```

## Roadmap
- Object tracking for accurate per-object screen-time (not just per-frame presence).
- Audio track analysis (speech/music/silence segmentation).
- A C++ (pybind11) accelerated frame-difference core for scene detection.

---
*Built as a learning project for computer-vision + media content analysis.*
