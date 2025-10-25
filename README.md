# Ad Intelligence Challenge – Feature Extraction Pipeline

A fast, modular pipeline to extract high-value, minimally overlapping signals from ad creatives (images and videos). Includes a CLI, parallel processing, and graceful fallbacks for optional models.

## Highlights
- Processes images (.png, .jpg, .jpeg) and videos (.mp4)
- Parallelizable via multiprocessing
- Under-5-minute target on sample set (machine-dependent)
- Modular extractors:
  - Core image: dimensions, aspect, color stats, brightness, saturation proxy, colorfulness, edge density
  - Core video: fps, duration, frame stats, motion intensity, shot changes, early-action ratio
  - Optional: OCR text area ratio (Tesseract or EasyOCR), CLIP embeddings (for semantic similarity and tone), audio MFCC/loudness
- Outputs CSV (default) or Parquet (if pyarrow installed)

## Quickstart

1) Create and activate a virtualenv (recommended)

```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install deps

```
pip install -r requirements.txt
```

3) Run the CLI on a zip or directory

```
python scripts/process_ads.py \
  --input /path/to/ads.zip \
  --output outputs/features.csv \
  --format csv \
  --workers 4
```

Or process a directory:

```
python scripts/process_ads.py --input /path/to/ads_dir --output outputs/features.parquet --format parquet
```

4) Optional features
- OCR: install one of: `pip install easyocr` (no external binary), or use Tesseract (`brew install tesseract && pip install pytesseract`).
- CLIP: `pip install open_clip_torch torch torchvision`.
- Audio: `pip install moviepy librosa soundfile`.

The pipeline will auto-detect installed optional deps and add corresponding features.

## Signals and Rationale
- Distinct, low-correlation core features: geometry (size/aspect), color stats vs. edges vs. motion.
- Predictive intuition examples:
  - Strong early motion can help retain attention in feed environments.
  - High edge density can correlate with busier creatives—may affect comprehension.
  - Colorfulness and saturation relate to salience.
  - OCR text ratio suggests presence of CTAs or overlays.
  - CLIP embeddings enable semantic similarity (tone, category, brand style) for retrieval-based modeling.

## Architecture
- `ad_intel/extractors/`: pluggable modules for image/video and optional features.
- `ad_intel/pipeline.py`: routing, parallel execution, robust error handling.
- `scripts/process_ads.py`: CLI and batch orchestration.

## Output Schema (core subset)
- Common: `id`, `media_type`, `error`
- Image: `width`, `height`, `aspect_ratio`, `mean_r/g/b`, `std_r/g/b`, `brightness`, `saturation_proxy`, `colorfulness`, `edge_density`, `text_area_ratio(opt)`, `clip_dim(opt)`
- Video: `width`, `height`, `fps`, `duration_sec`, `frame_count`, `avg_motion`, `shot_changes`, `early_action_ratio`, `audio_loudness(opt)`, `audio_tempo_bpm(opt)`, `clip_dim(opt)`

## Notes
- The pipeline logs errors per item and continues.
- Reproducibility: deterministic random seeds and fixed frame sampling intervals.

## License
MIT
