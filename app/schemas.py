"""API request/response models."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float
    box: List[int]  # [x, y, w, h]


class ColorInfo(BaseModel):
    hex: str
    proportion: float


class ImageAnalysis(BaseModel):
    width: int
    height: int
    detections: List[Detection] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    tag_counts: Dict[str, int] = Field(default_factory=dict)
    dominant_colors: List[ColorInfo] = Field(default_factory=list)
    annotated_image: Optional[str] = None  # data URI (base64 JPEG)


class TimelineEntry(BaseModel):
    time_sec: float
    tags: List[str]


class VideoAnalysis(BaseModel):
    duration_sec: float
    fps: float
    frames_analyzed: int
    tag_counts: Dict[str, int] = Field(default_factory=dict)
    screen_time_sec: Dict[str, float] = Field(default_factory=dict)
    timeline: List[TimelineEntry] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    detector: str
