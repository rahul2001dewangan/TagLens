"""FastAPI application for TagLens."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .analytics import summarize_detections
from .config import get_settings
from .detectors import get_detector
from .imaging import annotate, dominant_colors, read_image, to_data_uri
from .schemas import (
    ColorInfo,
    Detection,
    HealthResponse,
    ImageAnalysis,
    TimelineEntry,
    VideoAnalysis,
)
from .video import analyze_video


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    logger = logging.getLogger("taglens")
    detector = get_detector(settings)

    app = FastAPI(
        title="TagLens",
        version="1.0.0",
        description="Auto media tagger: detect objects/faces in images and short videos.",
    )
    app.state.detector = detector
    web_index = Path(__file__).parent / "web" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index():
        return web_index.read_text(encoding="utf-8")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", detector=detector.name)

    @app.post("/analyze/image", response_model=ImageAnalysis)
    async def analyze_image(file: UploadFile = File(...), annotate_image: bool = True):
        data = await file.read()
        try:
            img = read_image(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        detections = detector.detect(img)
        summary = summarize_detections(detections)
        colors = dominant_colors(img, settings.dominant_colors_k)
        height, width = img.shape[:2]
        annotated = to_data_uri(annotate(img, detections)) if annotate_image else None
        logger.info("image analyzed: %s tags=%s", file.filename, summary["tags"])
        return ImageAnalysis(
            width=width,
            height=height,
            detections=[Detection(**d) for d in detections],
            tags=summary["tags"],
            tag_counts=summary["tag_counts"],
            dominant_colors=[ColorInfo(**c) for c in colors],
            annotated_image=annotated,
        )

    @app.post("/analyze/video", response_model=VideoAnalysis)
    async def analyze_video_endpoint(file: UploadFile = File(...)):
        data = await file.read()
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            result = analyze_video(tmp_path, detector, settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        finally:
            os.unlink(tmp_path)
        logger.info("video analyzed: %s frames=%s", file.filename, result["frames_analyzed"])
        return VideoAnalysis(
            duration_sec=result["duration_sec"],
            fps=result["fps"],
            frames_analyzed=result["frames_analyzed"],
            tag_counts=result["tag_counts"],
            screen_time_sec=result["screen_time_sec"],
            timeline=[TimelineEntry(**t) for t in result["timeline"]],
        )

    return app


app = create_app()
