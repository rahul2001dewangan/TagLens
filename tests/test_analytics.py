from app.analytics import aggregate_video, summarize_detections


def test_summarize_detections():
    dets = [
        {"label": "face", "confidence": 0.9, "box": [0, 0, 10, 10]},
        {"label": "face", "confidence": 0.8, "box": [5, 5, 10, 10]},
        {"label": "object", "confidence": 0.7, "box": [1, 1, 2, 2]},
    ]
    out = summarize_detections(dets)
    assert out["tags"] == ["face", "object"]
    assert out["tag_counts"] == {"face": 2, "object": 1}


def test_aggregate_video_screen_time():
    # fps=10, step=5 -> each analysed frame represents 0.5s
    frame_results = [
        (0, [{"label": "face"}]),
        (5, [{"label": "face"}, {"label": "car"}]),
        (10, []),
    ]
    out = aggregate_video(frame_results, fps=10.0, step=5)
    assert out["tag_counts"] == {"face": 2, "car": 1}
    # face seen in 2 frames -> 2 * 0.5 = 1.0s ; car in 1 -> 0.5s
    assert out["screen_time_sec"]["face"] == 1.0
    assert out["screen_time_sec"]["car"] == 0.5
    assert len(out["timeline"]) == 2  # empty frame contributes no timeline entry
