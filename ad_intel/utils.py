from __future__ import annotations
import math
from pathlib import Path
from typing import Optional

import numpy as np


def safe_mean_color(img_arr: np.ndarray) -> tuple[float, float, float]:
    # Expect HxWxC in RGB
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        return (np.nan, np.nan, np.nan)
    mean = img_arr.reshape(-1, 3).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def safe_std_color(img_arr: np.ndarray) -> tuple[float, float, float]:
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        return (np.nan, np.nan, np.nan)
    std = img_arr.reshape(-1, 3).std(axis=0)
    return float(std[0]), float(std[1]), float(std[2])


def colorfulness_hasler(img_arr: np.ndarray) -> float:
    # https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
    R, G, B = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def brightness_proxy(img_arr: np.ndarray) -> float:
    # Luma proxy from RGB
    R, G, B = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return float(np.mean(Y))


def saturation_proxy(img_arr: np.ndarray) -> float:
    # Approximate saturation = std over channels
    return float(np.mean(np.std(img_arr.astype(np.float32), axis=2)))


def aspect_ratio(w: int, h: int) -> float:
    return float(w) / float(h) if h else math.nan


def try_import(module: str) -> Optional[object]:
    try:
        return __import__(module)
    except Exception:
        return None
