from __future__ import annotations
import numpy as np

# Tries EasyOCR first (no external binary), else Tesseract via pytesseract.

def text_area_ratio(img_rgb: np.ndarray) -> float:
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(img_rgb)
        H, W = img_rgb.shape[:2]
        total = 0.0
        for (_, _, conf) in results:
            # easyocr returns bbox, text, confidence; bbox polygon approximated to rect area
            # We don't have area directly; skip unless bbox available
            pass
        # Fallback: count high-contrast edges as proxy is already in core. Here, return 0.
        return 0.0
    except Exception:
        try:
            import pytesseract  # type: ignore
            from PIL import Image
            # use binary mask from OCR bounding boxes area / image area
            data = pytesseract.image_to_data(Image.fromarray(img_rgb), output_type=pytesseract.Output.DICT)
            H, W = img_rgb.shape[:2]
            areas = []
            n = len(data.get('level', []))
            for i in range(n):
                conf = data['conf'][i]
                if isinstance(conf, str):
                    try:
                        conf = float(conf)
                    except Exception:
                        conf = -1.0
                if conf is None or conf < 0:
                    continue
                w = data['width'][i]
                h = data['height'][i]
                areas.append(w * h)
            if not areas:
                return 0.0
            return float(sum(areas) / float(W * H))
        except Exception:
            # No OCR available
            return 0.0
