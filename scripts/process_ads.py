#!/usr/bin/env python3
import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ad_intel.pipeline import process_paths_parallel, detect_media_type


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    return dest_dir


def find_image_paths() -> list[Path]:
    """Find all image files in the images directory."""
    images_dir = Path("/Users/navitraa/ad-intelligence/inputs/images")
    image_exts = {'.png', '.jpg', '.jpeg'}
    
    paths = []
    if images_dir.exists():
        for root, _, files in os.walk(images_dir):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in image_exts:
                    paths.append(p)
    return sorted(paths)


def find_video_paths() -> list[Path]:
    """Find all video files in the videos directory."""
    videos_dir = Path("/Users/navitraa/ad-intelligence/inputs/videos")
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    
    paths = []
    if videos_dir.exists():
        for root, _, files in os.walk(videos_dir):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in video_exts:
                    paths.append(p)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Ad Intelligence Feature Extraction")
    parser.add_argument('--input', required=True, type=Path, help='Path to ads.zip or directory')
    parser.add_argument('--output', required=True, type=Path, help='Output file path (.csv or .parquet)')
    parser.add_argument('--format', choices=['csv', 'parquet'], default='csv')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--frame-interval', type=float, default=0.5, help='Seconds between sampled frames for video features')
    parser.add_argument('--max-frames', type=int, default=120, help='Max frames to sample per video')
    args = parser.parse_args()

    # Get image and video paths from separate directories
    image_paths = find_image_paths()
    video_paths = find_video_paths()
    
    if not image_paths and not video_paths:
        print("No media found in images or videos directories.")
        sys.exit(1)

    items = []
    
    # Add image files
    for p in image_paths:
        items.append({
            'id': p.stem,
            'path': str(p),
            'media_type': 'image'
        })
    
    # Add video files
    for p in video_paths:
        items.append({
            'id': p.stem,
            'path': str(p),
            'media_type': 'video'
        })

    results = process_paths_parallel(
        items,
        workers=args.workers,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
    )

    df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'csv' or args.output.suffix.lower() == '.csv':
        df.to_csv(args.output, index=False)
    else:
        try:
            import pyarrow  # noqa: F401
            df.to_parquet(args.output, index=False)
        except Exception as e:
            print(f"Parquet dependencies missing, writing CSV instead: {e}")
            csv_fallback = args.output.with_suffix('.csv')
            df.to_csv(csv_fallback, index=False)
            print(f"Wrote CSV to {csv_fallback}")

    print(f"Processed {len(df)} items. Output: {args.output}")


if __name__ == '__main__':
    main()
