"""Settings, loaded from environment variables or a .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Detector backend
    detector: str = "objects"       # objects (80 COCO classes) | yunet (faces) | stub | yolo
    object_model: str = "models/object_detection_yolox.onnx"
    yunet_model: str = "models/face_detection_yunet.onnx"
    yolo_model: str = "yolov8n.pt"
    min_confidence: float = 0.5

    # Video analysis
    video_frame_step: int = 15      # analyse every Nth frame
    max_frames: int = 300           # safety cap on frames analysed

    # Image
    dominant_colors_k: int = 4

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
