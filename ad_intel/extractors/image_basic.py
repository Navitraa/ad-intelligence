from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from ..utils import (
    safe_mean_color,
    safe_std_color,
    colorfulness_hasler,
    brightness_proxy,
    saturation_proxy,
    aspect_ratio,
)


def edge_density(img_arr: np.ndarray) -> float:
    # Expect RGB 0-255
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.mean(edges > 0))


def extract_image_features(path: Path) -> dict:
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        arr = np.array(im)

    mean_r, mean_g, mean_b = safe_mean_color(arr)
    std_r, std_g, std_b = safe_std_color(arr)

    feats = {
        'width': int(w),
        'height': int(h),
        'aspect_ratio': aspect_ratio(w, h),
        'mean_r': mean_r,
        'mean_g': mean_g,
        'mean_b': mean_b,
        'std_r': std_r,
        'std_g': std_g,
        'std_b': std_b,
        'brightness': brightness_proxy(arr),
        'saturation_proxy': saturation_proxy(arr),
        'colorfulness': colorfulness_hasler(arr),
        'edge_density': edge_density(arr),
    }

    # Optional OCR text area ratio
    try:
        from .ocr_optional import text_area_ratio
        feats['text_area_ratio'] = float(text_area_ratio(arr))
    except Exception:
        pass

    # Optional CLIP embedding dimensionality (not the vector to keep CSV small)
    try:
        from .clip_optional import clip_embed_dim
        feats['clip_dim'] = int(clip_embed_dim())
    except Exception:
        pass

    return feats
