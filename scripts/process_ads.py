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


def find_media_paths(input_path: Path) -> list[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.mp4'}
    if input_path.is_file():
        if input_path.suffix.lower() == '.zip':
            tmp = input_path.parent / (input_path.stem + "_extracted")
            extracted = extract_zip(input_path, tmp)
            search_root = extracted
        else:
            return [input_path]
    else:
        search_root = input_path

    paths = []
    for root, _, files in os.walk(search_root):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
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

    media_paths = find_media_paths(args.input)
    if not media_paths:
        print("No media found.")
        sys.exit(1)

    items = []
    for p in media_paths:
        items.append({
            'id': p.stem,
            'path': str(p),
            'media_type': detect_media_type(p.suffix.lower())
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
