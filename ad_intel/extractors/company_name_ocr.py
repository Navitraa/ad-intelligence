from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
import re


def extract_company_name_top_left(img_rgb: np.ndarray, region_fraction: float = 0.3) -> Optional[str]:
    """
    Extract company name from the top-left region of an image.
    
    Args:
        img_rgb: RGB image array
        region_fraction: Fraction of image width/height to consider as "top-left" (default 0.3 = 30%)
    
    Returns:
        Company name string or None if not found
    """
    H, W = img_rgb.shape[:2]
    
    # Define top-left region
    top_left_h = int(H * region_fraction)
    top_left_w = int(W * region_fraction)
    
    # Crop to top-left region
    top_left_region = img_rgb[:top_left_h, :top_left_w]
    
    # Try EasyOCR first
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(top_left_region)
        
        # Extract text with confidence > 0.5
        texts = []
        for (bbox, text, conf) in results:
            if conf > 0.5:
                texts.append(text.strip())
        
        # Find potential company name
        company_name = _find_company_name_from_texts(texts)
        if company_name:
            return company_name
            
    except Exception:
        pass
    
    # Fallback to Tesseract
    try:
        import pytesseract
        from PIL import Image
        
        # Convert to PIL Image
        pil_img = Image.fromarray(top_left_region)
        
        # Extract text
        text = pytesseract.image_to_string(pil_img, config='--psm 6')
        
        # Clean and find company name
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        company_name = _find_company_name_from_texts(lines)
        if company_name:
            return company_name
            
    except Exception:
        pass
    
    return None


def _find_company_name_from_texts(texts: List[str]) -> Optional[str]:
    """
    Find the most likely company name from extracted text lines.
    
    Heuristics:
    - Look for capitalized words
    - Avoid common words like "THE", "AND", "OF"
    - Prefer shorter, brand-like names
    - Skip URLs, emails, phone numbers
    """
    if not texts:
        return None
    
    # Common words to ignore
    ignore_words = {'THE', 'AND', 'OF', 'FOR', 'WITH', 'BY', 'IN', 'ON', 'AT', 'TO', 'FROM'}
    
    candidates = []
    
    for text in texts:
        # Clean text
        text = re.sub(r'[^a-zA-Z0-9\s&.-]', '', text)
        text = text.strip()
        
        if not text or len(text) < 2:
            continue
            
        # Skip if looks like URL, email, or phone
        if any(pattern in text.lower() for pattern in ['.com', '.net', '.org', '@', 'www.', 'http']):
            continue
            
        if re.match(r'^[\d\s\-\(\)\+]+$', text):  # Phone number pattern
            continue
            
        # Look for company-like patterns
        words = text.split()
        
        # Single word companies (e.g., "Nike", "Apple")
        if len(words) == 1 and len(text) >= 3 and text.isupper():
            candidates.append((text, 3))  # High priority
            
        # Multi-word companies (e.g., "Coca Cola", "General Motors")
        elif len(words) <= 4:
            # Filter out common words
            filtered_words = [w for w in words if w.upper() not in ignore_words]
            if filtered_words:
                company_text = ' '.join(filtered_words)
                # Prefer if most words are capitalized
                cap_ratio = sum(1 for w in filtered_words if w[0].isupper()) / len(filtered_words)
                priority = 2 if cap_ratio > 0.5 else 1
                candidates.append((company_text, priority))
    
    if not candidates:
        return None
    
    # Sort by priority (higher first), then by length (shorter first)
    candidates.sort(key=lambda x: (-x[1], len(x[0])))
    
    return candidates[0][0]


def get_company_name_features(img_rgb: np.ndarray) -> dict:
    """
    Extract company name features for integration with main feature extraction.
    
    Returns:
        Dictionary with company name and related features
    """
    company_name = extract_company_name_top_left(img_rgb)
    
    return {
        'company_name': company_name or '',
        'has_company_name': bool(company_name),
        'company_name_length': len(company_name) if company_name else 0
    }
