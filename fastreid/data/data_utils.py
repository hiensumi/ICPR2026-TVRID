# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import os
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import threading
from scipy.ndimage import uniform_filter1d

import queue
from torch.utils.data import DataLoader

from fastreid.utils.file_io import PathManager


def depth_foreground_mask(depth_path, dataset_hint='auto'):
    """Create a binary foreground mask from a depth image (v5).

    Auto-detects scene type from spatial depth pattern:
      - DB (flat floor + wooden passage): person-relative midpoint threshold,
        center-biased component selection, morphological refinement.
      - TVPR/TVPR2 (desks at top/bottom): desk-band exclusion (top/bottom 22%),
        strict+loose marker matching, safety cap at 22%.

    Args:
        depth_path: path to uint16 depth PNG
        dataset_hint: 'auto' (default), 'db', or 'tvpr'
    Returns:
        mask (np.ndarray, bool): H×W mask, True = foreground (person), or None
    """
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        return None

    depth = raw.astype(np.float32)
    h, w = depth.shape
    valid = depth > 0
    valid_vals = depth[valid]

    if len(valid_vals) < 200:
        return None

    # Floor detection — dominant background surface in upper 50%
    upper_vals = valid_vals[valid_vals >= np.percentile(valid_vals, 50)]
    hist_c, hist_e = np.histogram(upper_vals, bins=50)
    peak = np.argmax(hist_c)
    floor_depth = (hist_e[peak] + hist_e[peak + 1]) / 2.0

    # Auto-detect: TVPR has desks (top/bottom shallow, middle deep)
    if dataset_hint == 'auto':
        bh = h // 3
        top_v = depth[:bh, :]; top_v = top_v[top_v > 0]
        mid_v = depth[bh:2*bh, :]; mid_v = mid_v[mid_v > 0]
        top_med = np.median(top_v) if len(top_v) > 0 else floor_depth
        mid_med = np.median(mid_v) if len(mid_v) > 0 else floor_depth
        dataset_hint = 'tvpr' if (mid_med - top_med > 300) else 'db'

    kern5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kern7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kern11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    if dataset_hint == 'db':
        # ── DB: person-relative midpoint threshold ──
        # Find person depth peak from full histogram
        ahc, ahe = np.histogram(valid_vals, bins=100)
        sm = uniform_filter1d(ahc.astype(float), size=5)
        pb = 0
        for i in range(len(sm)):
            if sm[i] > sm.max() * 0.1:
                pb = i
                break
        person_d = (ahe[pb] + ahe[pb + 1]) / 2.0
        gap = floor_depth - person_d
        midp = person_d + gap * 0.5

        mask = (valid & (depth <= midp)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern7, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern5, iterations=1)

        # Center-biased component selection
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if num <= 1:
            return None
        cy_img, cx_img = h / 2.0, w / 2.0
        max_dist = np.sqrt(cy_img**2 + cx_img**2)
        best_lbl, best_score = -1, -1
        for lbl in range(1, num):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 0.003 * h * w:
                continue
            cx, cy = centroids[lbl]
            dist = np.sqrt((cx - cx_img)**2 + (cy - cy_img)**2)
            proximity = 1.0 - (dist / max_dist)
            score = np.sqrt(area) * (proximity ** 2)
            if score > best_score:
                best_score = score
                best_lbl = lbl
        if best_lbl < 0:
            return None

        body = (labels == best_lbl).astype(np.uint8) * 255
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kern11, iterations=2)
        body = cv2.morphologyEx(body, cv2.MORPH_DILATE, kern5, iterations=1)
        return body > 0

    else:
        # ── TVPR/TVPR2: desk-band exclusion + marker matching ──
        desk_band = int(h * 0.22)

        # Strict mask: floor - 400mm, exclude desk bands
        strict_thresh = floor_depth - 400
        strict_mask = (valid & (depth <= strict_thresh)).astype(np.uint8) * 255
        strict_mask[:desk_band, :] = 0
        strict_mask[h - desk_band:, :] = 0
        strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, kern7, iterations=2)
        strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, kern5, iterations=1)

        # Center-biased component from strict
        num_s, labels_s, stats_s, centroids_s = cv2.connectedComponentsWithStats(strict_mask, 8)
        if num_s <= 1:
            return None
        cy_img, cx_img = h / 2.0, w / 2.0
        max_dist = np.sqrt(cy_img**2 + cx_img**2)
        best_lbl, best_score = -1, -1
        for lbl in range(1, num_s):
            area = stats_s[lbl, cv2.CC_STAT_AREA]
            if area < 0.003 * h * w:
                continue
            cx, cy = centroids_s[lbl]
            dist = np.sqrt((cx - cx_img)**2 + (cy - cy_img)**2)
            proximity = 1.0 - (dist / max_dist)
            score = np.sqrt(area) * (proximity ** 2)
            if score > best_score:
                best_score = score
                best_lbl = lbl
        if best_lbl < 0:
            return None
        person_marker = (labels_s == best_lbl)

        # Loose mask: floor - 100mm, exclude desk bands
        loose_thresh = floor_depth - 100
        loose_mask = (valid & (depth <= loose_thresh)).astype(np.uint8) * 255
        loose_mask[:desk_band, :] = 0
        loose_mask[h - desk_band:, :] = 0
        loose_mask = cv2.morphologyEx(loose_mask, cv2.MORPH_CLOSE, kern7, iterations=3)
        loose_mask = cv2.morphologyEx(loose_mask, cv2.MORPH_OPEN, kern5, iterations=1)

        # Match loose component overlapping with person marker
        num_l, labels_l, _, _ = cv2.connectedComponentsWithStats(loose_mask, 8)
        if num_l > 1:
            best_l, best_ov = -1, 0
            for lbl in range(1, num_l):
                ov = np.sum(person_marker & (labels_l == lbl))
                if ov > best_ov:
                    best_ov = ov
                    best_l = lbl
            if best_l >= 0:
                body = (labels_l == best_l).astype(np.uint8) * 255
            else:
                body = (person_marker * 255).astype(np.uint8)
        else:
            body = (person_marker * 255).astype(np.uint8)

        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kern11, iterations=2)
        body = cv2.morphologyEx(body, cv2.MORPH_DILATE, kern5, iterations=1)

        result = body > 0
        # Safety cap: if mask > 22% of image, fall back to strict marker only
        if result.sum() / result.size > 0.22:
            body2 = (person_marker * 255).astype(np.uint8)
            body2 = cv2.morphologyEx(body2, cv2.MORPH_CLOSE, kern11, iterations=2)
            body2 = cv2.morphologyEx(body2, cv2.MORPH_DILATE, kern5, iterations=2)
            return body2 > 0
        return result


# Cache: directory → sorted list of (timestamp_int, depth_filename)
_depth_dir_cache = {}


def _find_depth_for_rgb(rgb_path):
    """Find the co-located depth file for an RGB image.
    
    Handles two naming conventions:
      - DB:   <timestamp>_RGB.png  →  <timestamp>_depth.png  (same timestamp)
      - TVPR: <ts_rgb>_RGB.png  →  nearest <ts_depth>_depth.png  (different timestamp)
    
    Uses a per-directory cache for efficient TVPR lookups.
    Returns depth_path or None.
    """
    # Try direct replacement first (works for DB data)
    direct = rgb_path.replace('_RGB.png', '_depth.png')
    if os.path.exists(direct):
        return direct
    
    # For TVPR: find nearest depth file in same directory by timestamp
    dirname = os.path.dirname(rgb_path)
    basename = os.path.basename(rgb_path)
    rgb_ts_str = basename.replace('_RGB.png', '')
    try:
        rgb_ts = int(rgb_ts_str)
    except ValueError:
        return None
    
    # Build/use cached sorted list of depth timestamps for this directory
    if dirname not in _depth_dir_cache:
        depth_files = []
        try:
            for fname in os.listdir(dirname):
                if fname.endswith('_depth.png'):
                    ts_str = fname.replace('_depth.png', '')
                    try:
                        depth_files.append((int(ts_str), fname))
                    except ValueError:
                        continue
        except OSError:
            _depth_dir_cache[dirname] = []
            return None
        depth_files.sort()
        _depth_dir_cache[dirname] = depth_files
    
    depth_files = _depth_dir_cache[dirname]
    if not depth_files:
        return None
    
    # Binary search for nearest depth timestamp
    import bisect
    timestamps = [t for t, _ in depth_files]
    idx = bisect.bisect_left(timestamps, rgb_ts)
    
    best_idx = None
    best_diff = float('inf')
    for candidate in (idx - 1, idx):
        if 0 <= candidate < len(timestamps):
            diff = abs(timestamps[candidate] - rgb_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = candidate
    
    if best_idx is not None:
        return os.path.join(dirname, depth_files[best_idx][1])
    return None


def read_image_depth_masked(rgb_path, fill_value=(127, 127, 127)):
    """Read an RGB image and mask out background using co-located depth.
    
    Derives depth path from RGB path, handling both DB and TVPR naming conventions.
    Background pixels are replaced with fill_value (training pixel mean).
    Falls back to normal read_image if depth is unavailable.
    
    Args:
        rgb_path: path to the RGB image
        fill_value: tuple (R, G, B) to fill background pixels
    Returns:
        PIL Image (RGB)
    """
    # Find depth file (handles both DB and TVPR naming)
    depth_path = _find_depth_for_rgb(rgb_path)
    
    # Try to get foreground mask from depth
    mask = None
    if depth_path is not None and os.path.exists(depth_path):
        mask = depth_foreground_mask(depth_path)
    
    # Load RGB normally
    image = read_image(rgb_path)  # returns PIL Image
    
    if mask is not None:
        img_np = np.asarray(image).copy()
        # Resize mask to match RGB if needed (should be same size for RealSense)
        if mask.shape[:2] != img_np.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img_np.shape[1], img_np.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        # Apply mask: keep person, fill background
        img_np[~mask] = fill_value
        image = Image.fromarray(img_np)
    
    return image


# ============================================================================
#  Body-Part Random Erasing (BPE) — uses precomputed depth masks
# ============================================================================

# Map from source image dirs to mask dirs (auto-detected)
_BPE_MASK_MAP = None

def _build_mask_path_map():
    """Build mapping from source image dirs to precomputed mask dirs."""
    global _BPE_MASK_MAP
    if _BPE_MASK_MAP is not None:
        return _BPE_MASK_MAP

    # TVRID layout:   data/<src>_extracted/.../xxx_RGB.png → data/<src>_extracted_SAM31_masks/.../xxx_mask.png
    # Market-1501:    .../market1501/<split>/xxx.jpg      → .../market1501_masks/<split>/xxx_mask.png
    #   (sibling dir because market1501/ is root-owned on the NFS share)
    _BPE_MASK_MAP = [
        ('DB_extracted', 'DB_extracted_SAM31_masks'),
        ('TVPR_2_extracted', 'TVPR_2_extracted_SAM31_masks'),
        ('TVPR_extracted', 'TVPR_extracted_SAM31_masks'),
        # Fallback for older depth-guided masks.
        ('DB_extracted', 'DB_extracted_masks'),
        ('TVPR_2_extracted', 'TVPR_2_extracted_masks'),
        ('TVPR_extracted', 'TVPR_extracted_masks'),
        # Market-1501: handled separately via MARKET1501_MASK_ROOT below
    ]
    return _BPE_MASK_MAP


# Mask roots relative to project
_DATASETS_ROOT = Path(__file__).resolve().parent.parent.parent / "datasets"
_MARKET1501_MASK_ROOTS = [
    str(_DATASETS_ROOT / "market1501_sam31_masks"),
    str(_DATASETS_ROOT / "market1501_masks"),
]
_MARKET1501_SPLITS = {'bounding_box_train', 'bounding_box_test', 'query'}
_CUHK03_MASK_ROOT = str(_DATASETS_ROOT / "cuhk03_masks")
_CUHK03_SPLITS = {'images_detected', 'images_labeled'}
_MSMT17_MASK_ROOT = str(_DATASETS_ROOT / "msmt17_masks")
_MSMT17_MARKERS = ['/MSMT17_V1/', '/MSMT17_V2/']
_MSMT17_V2_DIR_MAP = {'mask_train_v2': 'train', 'mask_test_v2': 'test'}


def load_precomputed_mask(rgb_path):
    """Load precomputed binary mask for an RGB image.

    Supports three naming conventions:
      TVRID:       data/<src>_extracted/.../xxx_RGB.png → <src>_extracted_masks/.../xxx_mask.png
      Market-1501: .../market1501/<split>/xxx.jpg       → market1501_sam31_masks/<split>/xxx_mask.png
                                                         (fallback: market1501_masks/<split>/xxx_mask.png)
      CUHK03:      .../cuhk03/<split>/xxx.png           → cuhk03_masks/<split>/xxx_mask.png

    Returns:
        np.ndarray (H, W) of bool, True=person; or None if mask file not found
    """
    # ── Market-1501 fast path ──
    for split in _MARKET1501_SPLITS:
        marker = '/' + split + '/'
        idx = rgb_path.find(marker)
        if idx >= 0:
            filename = rgb_path[idx + len(marker):]
            stem = os.path.splitext(filename)[0]
            for mask_root in _MARKET1501_MASK_ROOTS:
                mask_path = os.path.join(mask_root, split, stem + '_mask.png')
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        return mask_img > 127
            return None

    # ── CUHK03 fast path ──
    for split in _CUHK03_SPLITS:
        marker = '/' + split + '/'
        idx = rgb_path.find(marker)
        if idx >= 0:
            filename = rgb_path[idx + len(marker):]
            stem = os.path.splitext(filename)[0]
            mask_path = os.path.join(_CUHK03_MASK_ROOT, split, stem + '_mask.png')
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    return mask_img > 127
            return None

    # ── MSMT17 fast path ──
    # V1: .../MSMT17_V1/train/0000/xxx.jpg → msmt17_masks/train/0000/xxx_mask.png
    # V2: .../MSMT17_V2/mask_train_v2/0000/xxx.jpg → msmt17_masks/train/0000/xxx_mask.png
    for marker in _MSMT17_MARKERS:
        idx = rgb_path.find(marker)
        if idx >= 0:
            rel = rgb_path[idx + len(marker):]  # e.g. train/0000/xxx.jpg or mask_train_v2/0000/xxx.jpg
            # V2 uses mask_train_v2/mask_test_v2 dirs — remap to train/test
            for v2_dir, v1_dir in _MSMT17_V2_DIR_MAP.items():
                if rel.startswith(v2_dir + '/'):
                    rel = v1_dir + rel[len(v2_dir):]
                    break
            stem = os.path.splitext(rel)[0]
            mask_path = os.path.join(_MSMT17_MASK_ROOT, stem + '_mask.png')
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    return mask_img > 127
            return None

    # ── TVRID path ──
    mapping = _build_mask_path_map()
    for src_dir, mask_dir in mapping:
        idx = rgb_path.find('/' + src_dir + '/')
        if idx >= 0:
            prefix = rgb_path[:idx + 1]
            rel    = rgb_path[idx + 1 + len(src_dir) + 1:]
            if rel.endswith('_RGB.png'):
                rel_stem = rel[:-len('_RGB.png')]
            else:
                rel_stem = os.path.splitext(rel)[0]
            mask_path = os.path.join(prefix + mask_dir, rel_stem + '_mask.png')
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    return mask_img > 127
            return None

    return None


def apply_body_part_erasing(img_pil, mask, erase_prob=0.5, erase_max=3,
                            sl=0.02, sh=0.4, r1=0.3,
                            fill_value=(127, 127, 127)):
    """REA-on-body erasing with original background restore.

    Steps:
      1. Save the original image (with natural background).
      2. Place 1..erase_max random erasing rectangles within the person
         bounding box — only person pixels inside each rectangle are filled.
      3. Background pixels stay untouched (original scene).

    This forces the model to learn partial body features while keeping
    scene context, complementing BGE which does the opposite.

    Args:
        img_pil: PIL Image (RGB)
        mask: bool ndarray (H, W), True = person
        erase_prob: probability of applying this augmentation
        erase_max: maximum number of erasing rectangles
        sl: min erasing area ratio (relative to person bbox)
        sh: max erasing area ratio (relative to person bbox)
        r1: min aspect ratio of erasing rectangle
        fill_value: RGB fill for erased person pixels

    Returns:
        PIL Image with person parts randomly erased, background intact
    """
    import random
    import math

    if random.random() > erase_prob:
        return img_pil

    if mask is None:
        return img_pil

    img_np = np.asarray(img_pil)
    h, w = img_np.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST) > 0

    # Find person bounding box
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img_pil
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    if bbox_h < 4 or bbox_w < 4:
        return img_pil

    bbox_area = bbox_h * bbox_w
    img_out = img_np.copy()

    n_erase = random.randint(1, erase_max)
    for _ in range(n_erase):
        # REA-style: try to find a valid rectangle within person bbox
        for _attempt in range(20):
            target_area = random.uniform(sl, sh) * bbox_area
            aspect = random.uniform(r1, 1.0 / r1)
            rh = int(round(math.sqrt(target_area * aspect)))
            rw = int(round(math.sqrt(target_area / aspect)))
            if rh >= bbox_h or rw >= bbox_w or rh < 1 or rw < 1:
                continue
            ry = y_min + random.randint(0, bbox_h - rh)
            rx = x_min + random.randint(0, bbox_w - rw)
            # Only erase person pixels inside rectangle
            rect_mask = mask[ry:ry+rh, rx:rx+rw]
            if rect_mask.any():
                img_out[ry:ry+rh, rx:rx+rw][rect_mask] = fill_value
                break

    return Image.fromarray(img_out)


def apply_background_erasing(img_pil, mask, erase_prob=0.5,
                             fill_value=(127, 127, 127)):
    """Random background erasing augmentation using a precomputed person mask.
    
    Replaces background pixels (floor, passage, furniture) with a fill value
    so the model focuses on person appearance rather than scene context.
    
    Args:
        img_pil: PIL Image (RGB)
        mask: bool ndarray (H, W), True = person
        erase_prob: probability of applying this augmentation
        fill_value: RGB fill for background pixels
    
    Returns:
        PIL Image with background erased
    """
    import random
    
    if random.random() > erase_prob:
        return img_pil
    
    if mask is None:
        return img_pil
    
    img_np = np.asarray(img_pil).copy()
    h, w = img_np.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST) > 0
    if mask.mean() < 0.005:
        # Empty/tiny masks are failed segmentations; otherwise this blanks the whole frame.
        return img_pil
    
    # Fill background (non-person) with fill_value
    img_np[~mask] = fill_value
    
    return Image.fromarray(img_np)


def apply_background_alternating(img_pil, mask, prob=0.5, contrast_thresh=80.0, 
                                 mode='random_color', specific_color=(0,0,0), noise_std=30.0):
    """Background Alternating Augmentation: replaces background with a contrasting random color,
    a specific color, or Gaussian noise.

    Args:
        img_pil: PIL Image (RGB)
        mask: bool ndarray (H, W), True = person
        prob: probability of applying this augmentation
        contrast_thresh: minimum Euclidean distance (0-441 for RGB), used for 'random_color'
        mode: 'random_color', 'specific_color', or 'gaussian_noise'
        specific_color: (R, G, B) tuple used for 'specific_color' mode
        noise_std: standard deviation for 'gaussian_noise' mode
    
    Returns:
        PIL Image with background replaced.
    """
    import random
    import math

    if random.random() > prob:
        return img_pil

    if mask is None:
        return img_pil

    img_np = np.asarray(img_pil).copy()
    h, w = img_np.shape[:2]
    
    # Ensure mask matches image size
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST) > 0
    if mask.mean() < 0.005:
        # Empty/tiny masks are failed segmentations; otherwise this corrupts the whole frame.
        return img_pil

    if mode == 'random_choice':
        mode = 'random_color' if random.random() < 0.5 else 'gaussian_noise'

    if mode == 'specific_color':
        # Use provided color directly
        best_color = tuple(specific_color)
        img_np[~mask] = best_color
        
    elif mode == 'gaussian_noise':
        # Generate Gaussian noise
        noise = np.random.normal(127, noise_std, img_np.shape).astype(np.float32)
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        
        # Apply noise to background
        img_np[~mask] = noise[~mask]
        
    else: # 'random_color' (default)
        # 1. Compute person's mean color
        person_pixels = img_np[mask]
        if len(person_pixels) == 0:
            return img_pil # Empty mask, skip
        
        person_mean = np.mean(person_pixels, axis=0) # [R, G, B] float

        # 2. Find contrasting background color
        best_color = (127, 127, 127) # Fallback grey
        
        for _ in range(10): # Try 10 times to find a contrasting color
            cand = np.random.randint(0, 256, size=3) # [R, G, B]
            dist = np.linalg.norm(person_mean - cand)
            if dist > contrast_thresh:
                best_color = tuple(cand.tolist())
                break
                
        # 3. Replace background
        img_np[~mask] = best_color
    
    return Image.fromarray(img_np)


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        # FIX: Delay conversion to allow custom depth handling to see raw uint16
        image_np = np.asarray(image)

        # Convert uint16 (depth) to uint8 -> JET ColorMap for RGB-like features
        # MUST be done BEFORE grayscale expansion since depth is 2D
        if image_np.dtype == np.uint16 and len(image_np.shape) == 2:
            # TVRID depth: values typically 0-5000mm (uint16)
            img_float = image_np.astype(np.float32)
            
            valid_mask = img_float > 0
            valid_pixels = img_float[valid_mask]
            
            if len(valid_pixels) > 100:
                # Percentile normalization to handle outliers
                p_low = np.percentile(valid_pixels, 2)
                p_high = np.percentile(valid_pixels, 98)
                
                img_norm = np.clip(img_float, p_low, p_high)
                img_norm = (img_norm - p_low) / (p_high - p_low + 1e-6) * 255.0
                
                # INVERT: Close objects (low depth) become HIGH value (red in JET)
                # Far objects become LOW value (blue in JET)
                img_norm = 255.0 - img_norm
                
                # Zero out invalid pixels
                img_norm = np.where(valid_mask, img_norm, 0)
            else:
                img_norm = np.zeros_like(img_float)
                
            img_uint8 = img_norm.astype(np.uint8)
            
            # Apply JET colormap: 1 channel -> 3 channels (BGR)
            # Close person: Red/warm, Far background: Blue/cool
            img_color = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
            
            # JET maps 0 to dark blue [128,0,0 BGR]. Override to black for invalid.
            black_mask = (img_uint8 == 0) & (~valid_mask)
            img_color[black_mask] = [0, 0, 0]
            
            # BGR to RGB
            image = Image.fromarray(img_color[:, :, ::-1])

        else:
            if format is not None:
                # PIL only supports RGB, so convert to RGB and flip channels over below
                conversion_format = format
                if format == "BGR":
                    conversion_format = "RGB"
                image = image.convert(conversion_format)
            image_np = np.asarray(image)

            # PIL squeezes out the channel dimension for "L", so make it HWC
            if format == "L":
                image_np = np.expand_dims(image_np, -1)

            # handle formats not supported by PIL
            elif format == "BGR":
                # flip channels if needed
                image_np = image_np[:, :, ::-1]

            # handle grayscale mixed in RGB images
            elif len(image_np.shape) == 2:
                image_np = np.repeat(image_np[..., np.newaxis], 3, axis=-1)
                
            image = Image.fromarray(image_np)

        return image


"""
#based on http://stackoverflow.com/questions/7323664/python-generator-pre-fetch
This is a single-function package that transforms arbitrary generator into a background-thead generator that 
prefetches several batches of data in a parallel background thead.

This is useful if you have a computationally heavy process (CPU or GPU) that 
iteratively processes minibatches from the generator while the generator 
consumes some other resource (disk IO / loading from database / more CPU if you have unused cores). 

By default these two processes will constantly wait for one another to finish. If you make generator work in 
prefetch mode (see examples below), they will work in parallel, potentially saving you your GPU time.
We personally use the prefetch generator when iterating minibatches of data for deep learning with PyTorch etc.

Quick usage example (ipython notebook) - https://github.com/justheuristic/prefetch_generator/blob/master/example.ipynb
This package contains this object
 - BackgroundGenerator(any_other_generator[,max_prefetch = something])
"""


class BackgroundGenerator(threading.Thread):
    """
    the usage is below
    >> for batch in BackgroundGenerator(my_minibatch_iterator):
    >>    doit()
    More details are written in the BackgroundGenerator doc
    >> help(BackgroundGenerator)
    """

    def __init__(self, generator, local_rank, max_prefetch=10):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may raise GIL and zero-out the
        benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it
        outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving
        URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep
        stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until
        one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work
        slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size
        unless dequeued quickly enough.
        """
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(
            local_rank
        )  # create a new cuda stream in each process
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            # avoid re-entrance or ill-conditioned thread state
            return

        # Set exit event to True for background threading stopping
        self.iter.exit_event.set()

        # Exhaust all remaining elements, so that the queue becomes empty,
        # and the thread should quit
        for _ in self.iter:
            pass

        # Waiting for background thread to quit
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k].to(
                        device=self.local_rank, non_blocking=True
                    )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(
            self.stream
        )  # wait tensor to put on GPU
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        # If the dataloader is to be freed, shutdown its BackgroundGenerator
        self._shutdown_background_thread()
