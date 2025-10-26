from __future__ import annotations
from pathlib import Path

import numpy as np
import cv2

from ..utils import aspect_ratio


def _read_video_capture(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def _iter_sampled_frames(cap, frame_interval_sec: float, max_frames: int):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    step = max(int(round(frame_interval_sec * fps)), 1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    grabbed = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield frame
            grabbed += 1
            if grabbed >= max_frames:
                break
        idx += 1


def _motion_intensity(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    # Simple absolute difference norm as motion proxy
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))


def _shot_change(prev_hist: np.ndarray, hist: np.ndarray) -> bool:
    # Histogram correlation threshold for shot change
    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
    return bool(corr < 0.7)


def extract_video_features(path: Path, frame_interval: float, max_frames: int) -> dict:
    cap = _read_video_capture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = float(frame_count / fps) if fps > 0 else 0.0

    motions = []
    shot_changes = 0

    prev_gray = None
    prev_hist = None

    for frame_bgr in _iter_sampled_frames(cap, frame_interval, max_frames):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # motion
        if prev_gray is not None:
            motions.append(_motion_intensity(prev_gray, gray))
        prev_gray = gray

        # shot change via HSV histogram
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 50], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None and _shot_change(prev_hist, hist):
            shot_changes += 1
        prev_hist = hist

    cap.release()

    avg_motion = float(np.mean(motions)) if motions else 0.0

    # Early action ratio: motion in first 3 seconds vs overall
    cap2 = _read_video_capture(path)
    early_frames = max(int((fps or 30) * 3), 1)
    idx = 0
    motions_early = []
    prev_gray = None
    while idx < early_frames:
        ret, frame = cap2.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            motions_early.append(_motion_intensity(prev_gray, gray))
        prev_gray = gray
        idx += 1
    cap2.release()

    early_action_ratio = float(np.mean(motions_early) / avg_motion) if (motions_early and avg_motion > 1e-6) else 0.0

    feats = {
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio(width, height),
        'fps': fps,
        'duration_sec': duration_sec,
        'frame_count': frame_count,
        'avg_motion': avg_motion,
        'shot_changes': int(shot_changes),
        'early_action_ratio': early_action_ratio,
    }

    # Optional audio features
    try:
        from .audio_optional import extract_audio_features
        feats.update(extract_audio_features(path))
    except Exception:
        pass

    # Optional CLIP embedding dimensionality
    try:
        from .clip_optional import clip_embed_dim
        feats['clip_dim'] = int(clip_embed_dim())
    except Exception:
        pass

    # Audio transcript features (placeholder for future implementation)
    feats.update({
        'transcript': '',
        'has_speech': False,
        'word_count': 0,
        'sentence_count': 0,
        'avg_words_per_sentence': 0.0,
        'transcript_length': 0,
        'ad_keyword_count': 0,
        'ad_keyword_density': 0.0,
        'has_call_to_action': False,
    })

    return feats
