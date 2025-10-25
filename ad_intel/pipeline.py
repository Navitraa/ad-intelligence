from __future__ import annotations
import concurrent.futures as futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .extractors.image_basic import extract_image_features
from .extractors.video_basic import extract_video_features


MEDIA_IMAGE = {'png', 'jpg', 'jpeg'}
MEDIA_VIDEO = {'mp4'}


def detect_media_type(suffix: str) -> str:
    s = suffix.lstrip('.').lower()
    if s in MEDIA_IMAGE:
        return 'image'
    if s in MEDIA_VIDEO:
        return 'video'
    return 'unknown'


@dataclass
class WorkItem:
    id: str
    path: str
    media_type: str


def process_one(item: Dict[str, Any], frame_interval: float, max_frames: int) -> Dict[str, Any]:
    pid = item['id']
    p = Path(item['path'])
    media_type = item['media_type']
    out: Dict[str, Any] = {'id': pid, 'media_type': media_type}
    try:
        if media_type == 'image':
            out.update(extract_image_features(p))
        elif media_type == 'video':
            out.update(extract_video_features(p, frame_interval=frame_interval, max_frames=max_frames))
        else:
            out['error'] = 'unsupported_media'
    except Exception as e:
        out['error'] = str(e)
    return out


def process_paths_parallel(items: List[Dict[str, Any]], workers: int, frame_interval: float, max_frames: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(process_one, it, frame_interval, max_frames)
            for it in items
        ]
        for f in futures.as_completed(futs):
            results.append(f.result())
    return results
