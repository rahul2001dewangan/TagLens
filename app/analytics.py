"""Aggregate raw detections into tags, counts, and screen-time analytics."""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple


def summarize_detections(detections: List[dict]) -> dict:
    counts = Counter(d["label"] for d in detections)
    return {"tags": sorted(counts.keys()), "tag_counts": dict(counts)}


def aggregate_video(frame_results: List[Tuple[int, List[dict]]], fps: float, step: int) -> dict:
    """frame_results: list of (frame_index, detections_for_that_frame)."""
    tag_counts: Counter = Counter()      # total detections per label
    screen_frames: Counter = Counter()   # frames in which a label appears
    timeline = []
    for idx, dets in frame_results:
        labels = {d["label"] for d in dets}
        for d in dets:
            tag_counts[d["label"]] += 1
        for label in labels:
            screen_frames[label] += 1
        if labels:
            timeline.append({
                "time_sec": round(idx / fps, 2) if fps else 0.0,
                "tags": sorted(labels),
            })
    sec_per_sample = (step / fps) if fps else 0.0
    screen_time = {lbl: round(n * sec_per_sample, 2) for lbl, n in screen_frames.items()}
    return {
        "tag_counts": dict(tag_counts),
        "screen_time_sec": screen_time,
        "timeline": timeline,
    }
