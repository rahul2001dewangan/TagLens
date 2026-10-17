"""Short-video analysis: sample frames, detect, and aggregate analytics."""
from __future__ import annotations

from .analytics import aggregate_video


def analyze_video(path: str, detector, settings) -> dict:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video (unsupported or corrupt file)")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, settings.video_frame_step)

    frame_results = []
    idx = 0
    analyzed = 0
    while analyzed < settings.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame_results.append((idx, detector.detect(frame)))
            analyzed += 1
        idx += 1
    cap.release()

    if total_frames and fps:
        duration = round(total_frames / fps, 2)
    elif fps:
        duration = round(idx / fps, 2)
    else:
        duration = 0.0

    agg = aggregate_video(frame_results, fps, step)
    return {
        "duration_sec": duration,
        "fps": round(float(fps), 2),
        "frames_analyzed": analyzed,
        **agg,
    }
