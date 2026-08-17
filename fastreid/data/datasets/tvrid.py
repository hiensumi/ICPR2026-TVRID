# encoding: utf-8
"""
TVRID Dataset for ICPR 2026 Competition
Supports RGB, Depth, and Cross-modal tracks.
"""

import os
import os.path as osp
import csv
import hashlib
import json
import tempfile
from glob import glob
from pathlib import Path

import cv2
import numpy as np

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

__all__ = ['TVRID_RGB', 'TVRID_Depth', 'TVRID_Cross']


def load_csv_data(csv_path):
    """Load data from CSV file."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def get_best_frame(folder_path, modality='RGB', strategy='largest'):
    """Get the best frame from a folder of images.
    
    Args:
        folder_path: Path to folder containing frames
        modality: 'RGB' or 'depth'
        strategy: 'middle' (legacy), 'largest' (most foreground pixels for depth)
    
    Returns:
        Path to best frame or None if not found
    """
    suffix = f'_{modality}.png'
    pattern = osp.join(folder_path, f'*{suffix}')
    frames = sorted(glob(pattern))
    
    if not frames:
        return None
    
    if strategy == 'middle' or len(frames) == 1:
        mid_idx = len(frames) // 2
        return frames[mid_idx]
    
    if strategy == 'largest' and modality == 'depth':
        # Optimization: Use file size as proxy for content quantity (non-zero pixels).
        # Compressed PNGs with more depth data are significantly larger.
        # This avoids opening/decoding every single image (approx 100x faster).
        best_frame = None
        best_metric = -1
        for f in frames:
            try:
                # Use file size instead of reading image
                current_metric = os.path.getsize(f)
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_frame = f
            except Exception:
                continue
        return best_frame if best_frame else frames[len(frames) // 2]
    
    # Default: middle frame
    return frames[len(frames) // 2]


def get_middle_frame(folder_path, modality='RGB'):
    """Get the middle frame from a folder of images.
    
    Args:
        folder_path: Path to folder containing frames
        modality: 'RGB' or 'depth'
    
    Returns:
        Path to middle frame or None if not found
    """
    suffix = f'_{modality}.png'
    pattern = osp.join(folder_path, f'*{suffix}')
    frames = sorted(glob(pattern))
    
    if not frames:
        return None
    
    mid_idx = len(frames) // 2
    return frames[mid_idx]


def get_multiple_frames(folder_path, modality='depth', n_frames=5, strategy='top_n'):
    """Get multiple frames from a passage folder.
    
    Args:
        folder_path: Path to folder containing frames
        modality: 'RGB' or 'depth'
        n_frames: Number of frames to return
        strategy: 'top_n' (largest file size), 'top_half' (top half by file size),
            'half' (legacy alias for top_half),
            'uniform' (evenly spaced), 'all', 'middle_expand'
    
    Returns:
        List of frame paths (may be fewer than n_frames if not enough available)
    """
    suffix = f'_{modality}.png'
    pattern = osp.join(folder_path, f'*{suffix}')
    
    import re
    def extract_frame_num(path):
        filename = osp.basename(path)
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else path
        
    frames = sorted(glob(pattern), key=extract_frame_num)
    
    if not frames:
        return []
    
    if strategy == 'all':
        return frames
    
    if len(frames) <= n_frames:
        return frames
    
    if strategy == 'top_n':
        # Select frames with largest file size. 
        # For depth this means most non-zero pixels. For RGB it often correlates with subject size/detail.
        sized = [(os.path.getsize(f), f) for f in frames]
        sized.sort(key=lambda x: -x[0])  # Sort descending by size
        # Sort back temporally to keep sequential order
        return sorted([f for _, f in sized[:n_frames]])

    if strategy in ('top_half', 'half'):
        # Keep the top half by file size, capped by n_frames.
        # This gives variable-length passages a consistent "core" selection
        # without forcing every long passage to contribute the full cap.
        sized = [(os.path.getsize(f), f) for f in frames]
        sized.sort(key=lambda x: -x[0])  # Sort descending by size
        keep = min(n_frames, max(1, len(frames) // 2))
        return sorted([f for _, f in sized[:keep]])
    
    if strategy == 'middle_expand':
        # Select n_frames seamlessly centered around the middle frame
        mid = len(frames) // 2
        start = max(0, mid - n_frames // 2)
        end = start + n_frames
        # Readjust start if end exceeds length to guarantee N frames
        if end > len(frames):
            end = len(frames)
            start = max(0, end - n_frames)
        return frames[start:end]

    if strategy == 'uniform':
        # Evenly spaced frames including first and last
        indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        return [frames[i] for i in indices]
    
    # Default: evenly spaced
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    return [frames[i] for i in indices]


def get_core_frames(folder_path, modality='depth', ratio=0.5):
    """Get the 'core' frames (top % by file size) to avoid empty edge frames.
    
    In top-down videos, edge frames often only show empty ground or a shoe.
    For depth images, file size is a direct proxy for foreground pixels.
    This selects the top `ratio` largest frames and returns them.
    
    Args:
        folder_path: Path to folder containing frames
        modality: 'RGB' or 'depth'
        ratio: Fraction of frames to keep (e.g. 0.5 = keep top 50%)
    
    Returns:
        List of frame paths
    """
    suffix = f'_{modality}.png'
    pattern = osp.join(folder_path, f'*{suffix}')
    frames = sorted(glob(pattern))
    
    if not frames:
        return []
        
    if len(frames) <= 2:
        return frames
        
    if modality == 'depth':
        # Select frames with most depth content (largest file size = most non-zero pixels)
        sized = [(os.path.getsize(f), f) for f in frames]
        sized.sort(key=lambda x: -x[0])  # Sort descending by size
        
        keep_count = max(2, int(len(frames) * ratio))
        return [f for _, f in sized[:keep_count]]
        
    # Fallback for RGB: crop edges temporally
    keep_count = max(2, int(len(frames) * ratio))
    offset = (len(frames) - keep_count) // 2
    return frames[offset:offset + keep_count]


class TVRIDBase(ImageDataset):
    """Base class for TVRID datasets."""
    
    dataset_name = "tvrid"
    
    # Camera name to ID mapping for DB_extracted
    CAM_NAME_TO_ID = {
        'flat': 0,
        'upward': 1,
        'downward': 2,
        'upsideDown': 3,
        'topview': 4,  # TVPR_2
    }
    
    def __init__(self, root='datasets', **kwargs):
        # Allow overarching configs to dynamically hijack dataset class constants if provided
        if "n_frames" in kwargs:
            self.__class__.N_FRAMES = kwargs.pop("n_frames")
        if "frame_strategy" in kwargs:
            self.__class__.FRAME_STRATEGY = kwargs.pop("frame_strategy")

        # The 'root' from fast-reid defaults to 'datasets/', but we want to use '../data/'
        # Use absolute path for reliability
        self.root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))  # fast-reid root
        
        # Check if running on Kaggle
        if os.path.exists('/kaggle/input'):
            # Kaggle paths - try to find data in input
            # User specified path: /kaggle/input/reid-data/
            if os.path.exists('/kaggle/input/reid-data/DB_extracted'):
                self.db_dir = '/kaggle/input/reid-data/DB_extracted'
            elif os.path.exists('/kaggle/input/db-extracted'):
                self.db_dir = '/kaggle/input/db-extracted'
            else:
                self.db_dir = '/kaggle/input/tvrid-dataset/DB_extracted'  # Fallback guess
            
            if os.path.exists('/kaggle/input/reid-data/TVPR_2_extracted'):
                self.tvpr_dir = '/kaggle/input/reid-data/TVPR_2_extracted'
            elif os.path.exists('/kaggle/input/tvpr-2-extracted'):
                self.tvpr_dir = '/kaggle/input/tvpr-2-extracted'
            else:
                self.tvpr_dir = '/kaggle/input/tvrid-dataset/TVPR_2_extracted'
        else:
            # Local paths (relative to workspace root)
            self.db_dir = osp.join(self.root, 'data', 'DB_extracted')
            self.tvpr_dir = osp.join(self.root, 'data', 'TVPR_2_extracted')
            self.tvpr1_dir = osp.join(self.root, 'data', 'TVPR_extracted')
        
        # CSV files
        self.db_train_csv = osp.join(self.db_dir, 'train_labels.csv')
        self.tvpr_train_csv = osp.join(self.tvpr_dir, 'train_labels.csv')
        self.tvpr1_train_csv = osp.join(self.tvpr1_dir, 'train_labels.csv')
        self.db_test_csv = osp.join(self.db_dir, 'public_test_labels.csv')
        self.db_public_gt_dir = osp.join(self.db_dir, 'pubic_test_ground_truth')
        self.db_public_secret_map_csv = osp.join(self.db_public_gt_dir, 'test_secret_map.csv')
        
        # Check required files
        required = [self.db_dir, self.db_train_csv]
        if osp.exists(self.tvpr_dir):
            required.append(self.tvpr_train_csv)
        if osp.exists(self.tvpr1_dir):
            required.append(self.tvpr1_train_csv)
        self.check_before_run(required)
        
        # Process data
        train = lambda: self.process_train()
        query = lambda: self.process_test(mode='query')
        gallery = lambda: self.process_test(mode='gallery')
        
        super(TVRIDBase, self).__init__(train, query, gallery, **kwargs)
    
    def process_train(self):
        """Override in subclasses."""
        raise NotImplementedError
    
    def process_test(self, mode='query'):
        """Override in subclasses."""
        raise NotImplementedError


@DATASET_REGISTRY.register()
class TVRID_RGB(TVRIDBase):
    """TVRID RGB-only dataset for RGB track."""
    
    dataset_name = "tvrid_rgb"
    
    def process_train(self):
        data = []
        
        # Process DB_extracted
        db_data = load_csv_data(self.db_train_csv)
        for row in db_data:
            person_id = row['person_id']
            cam_name = row['cam_name']
            path = row['path'].replace('\\', '/')
            
            folder_path = osp.join(self.db_dir, 'train', path)
            img_path = get_middle_frame(folder_path, 'RGB')
            
            if img_path is None:
                continue
            
            cam_id = self.CAM_NAME_TO_ID.get(cam_name, 0)
            pid = f"{self.dataset_name}_{person_id}"
            camid = f"{self.dataset_name}_{cam_id}"
            
            data.append((img_path, pid, camid))
        
        # Process TVPR_2_extracted if available
        if osp.exists(self.tvpr_train_csv):
            tvpr_data = load_csv_data(self.tvpr_train_csv)
            for row in tvpr_data:
                person_id = row['person_id']
                cam_name = row['cam_name']
                path = row['path'].replace('\\', '/')
                
                folder_path = osp.join(self.tvpr_dir, 'train', path)
                img_path = get_middle_frame(folder_path, 'RGB')
                
                if img_path is None:
                    continue
                
                cam_id = self.CAM_NAME_TO_ID.get(cam_name, 4)
                # Offset person IDs to avoid collision with DB_extracted
                pid = f"{self.dataset_name}_tvpr_{person_id}"
                camid = f"{self.dataset_name}_{cam_id}"
                
                data.append((img_path, pid, camid))
        
        return data
    
    def process_test(self, mode='query'):
        """For test, all samples are both query and gallery."""
        data = []
        
        test_data = load_csv_data(self.db_test_csv)
        for i, row in enumerate(test_data):
            gallery_id = row['gallery_id']
            path = row['path'].replace('\\', '/')
            
            folder_path = osp.join(self.db_dir, 'test_public', path)
            img_path = get_middle_frame(folder_path, 'RGB')
            
            if img_path is None:
                continue
            
            # Use gallery_id as pid
            # CRITICAL: Use index as camid to filter SELF-MATCHES.
            # Market1501 metric excludes (SamePID & SameCam).
            # If Q and G are the same image, they will have SamePID and SameIndex (CamID).
            # So they are excluded.
            # Different images of same person -> SamePID, DiffIndex -> Kept.
            camid = i 
            data.append((img_path, gallery_id, camid))
        
        return data


@DATASET_REGISTRY.register()
class TVRID_Depth(TVRIDBase):
    """TVRID Depth-only dataset for Depth track."""
    
    dataset_name = "tvrid_depth"
    
    def process_train(self):
        data = []
        
        # Process DB_extracted
        db_data = load_csv_data(self.db_train_csv)
        for row in db_data:
            person_id = row['person_id']
            cam_name = row['cam_name']
            path = row['path'].replace('\\', '/')
            
            folder_path = osp.join(self.db_dir, 'train', path)
            img_path = get_best_frame(folder_path, 'depth', 'largest')
            
            if img_path is None:
                continue
            
            cam_id = self.CAM_NAME_TO_ID.get(cam_name, 0)
            pid = f"{self.dataset_name}_{person_id}"
            camid = f"{self.dataset_name}_{cam_id}"
            
            data.append((img_path, pid, camid))
        
        # Process TVPR_2_extracted if available
        if osp.exists(self.tvpr_train_csv):
            tvpr_data = load_csv_data(self.tvpr_train_csv)
            for row in tvpr_data:
                person_id = row['person_id']
                cam_name = row['cam_name']
                path = row['path'].replace('\\', '/')
                
                folder_path = osp.join(self.tvpr_dir, 'train', path)
                img_path = get_best_frame(folder_path, 'depth', 'largest')
                
                if img_path is None:
                    continue
                
                cam_id = self.CAM_NAME_TO_ID.get(cam_name, 4)
                pid = f"{self.dataset_name}_tvpr_{person_id}"
                camid = f"{self.dataset_name}_{cam_id}"
                
                data.append((img_path, pid, camid))
        
        return data
    
    def process_test(self, mode='query'):
        """For validation, use training data split since test set has no identity labels.
        
        Split training data into query (30%) and gallery (70%) for validation.
        Uses cam_id to distinguish query from gallery to filter self-matches.
        """
        data = []
        
        # Build a dict of person_id -> list of samples for proper splitting
        person_samples = {}
        db_data = load_csv_data(self.db_train_csv)
        
        for row in db_data:
            person_id = row['person_id']
            cam_name = row['cam_name']
            path = row['path'].replace('\\', '/')
            
            folder_path = osp.join(self.db_dir, 'train', path)
            img_path = get_best_frame(folder_path, 'depth', 'largest')
            
            if img_path is None:
                continue
            
            cam_id = self.CAM_NAME_TO_ID.get(cam_name, 0)
            pid = f"{self.dataset_name}_{person_id}"
            
            if pid not in person_samples:
                person_samples[pid] = []
            person_samples[pid].append((img_path, pid, cam_id, cam_name))
        
        # Only include persons with 2+ samples (can have both query and gallery)
        multi_sample_persons = {k: v for k, v in person_samples.items() if len(v) >= 2}
        
        for pid, samples in multi_sample_persons.items():
            if mode == 'query':
                # Take first sample as query, use cam_id 0
                img_path, pid_str, cam_id, cam_name = samples[0]
                data.append((img_path, pid_str, 0))
            else:  # gallery
                # Take remaining samples as gallery, use cam_id 1 to avoid self-match
                for img_path, pid_str, cam_id, cam_name in samples[1:]:
                    data.append((img_path, pid_str, 1))
        
        return data


@DATASET_REGISTRY.register()
class TVRID_Depth_CombinedSplit(TVRIDBase):
    """TVRID Depth-only dataset with combined sources and a leakage-free val split.

    - Training uses BOTH DB_extracted and TVPR_2_extracted.
    - Validation is identity-disjoint from training (reflects true ReID generalization).
    - Validation protocol: for each val identity, pick 1 query image and use the rest as gallery.
      Query camid=0, gallery camid=1 to avoid Market1501 filtering removing positives.

    Notes:
      This dataset is meant for *training-time validation*.
      For competition submission/public-test extraction, use the original datasets.
    """

    dataset_name = "tvrid_depth_combined"

    VAL_ID_RATIO = 0.2
    SPLIT_SEED = 42
    
    # Class-level cache for expensive _collect_depth_samples
    _pid_to_samples_cache = {}

    def _stable_int(self, s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}_r{self.VAL_ID_RATIO:.2f}.json",
        )

    def _load_or_create_val_pids(self, pid_to_samples):
        split_path = self._split_file()
        if osp.exists(split_path):
            try:
                with open(split_path, "r") as f:
                    obj = json.load(f)
                val_pids = set(obj.get("val_pids", []))
                if val_pids:
                    return val_pids
            except Exception:
                pass

        # Only identities with >=2 samples are eligible for validation (need query+gallery).
        candidates = sorted([pid for pid, samples in pid_to_samples.items() if len(samples) >= 2])
        if not candidates:
            return set()

        val_size = int(round(len(candidates) * float(self.VAL_ID_RATIO)))
        val_size = max(1, min(val_size, len(candidates) - 1)) if len(candidates) > 1 else 1

        # Deterministic shuffle
        rng = hashlib.md5(f"{self.dataset_name}:{self.SPLIT_SEED}".encode("utf-8")).hexdigest()
        # Use stable hash ordering with seed-derived salt
        salted = sorted(candidates, key=lambda pid: self._stable_int(pid + rng))
        val_pids = set(salted[:val_size])

        payload = {
            "dataset": self.dataset_name,
            "seed": self.SPLIT_SEED,
            "val_id_ratio": self.VAL_ID_RATIO,
            "num_ids_total": len(pid_to_samples),
            "num_ids_candidates": len(candidates),
            "num_ids_val": len(val_pids),
            "val_pids": sorted(val_pids),
        }
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=osp.dirname(split_path)) as tf:
                json.dump(payload, tf, indent=2)
                tmp_name = tf.name
            os.replace(tmp_name, split_path)
        except Exception:
            pass

        return val_pids

    def _collect_depth_samples(self):
        """Collect depth samples from both DB_extracted and TVPR_2_extracted.
        
        Uses class-level memory cache and JSON file cache to avoid re-scanning folders on every call.
        """
        import fcntl
        import os
        import json
        cache_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, f"{self.dataset_name}_single_frames_cache.json")
        lock_path = cache_path + ".lock"

        # Check in-memory cache first
        cache_key = self.dataset_name
        if cache_key in TVRID_Depth_CombinedSplit._pid_to_samples_cache:
            return TVRID_Depth_CombinedSplit._pid_to_samples_cache[cache_key]

        # Check JSON disk cache
        if osp.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    TVRID_Depth_CombinedSplit._pid_to_samples_cache[cache_key] = data
                    return data
            except json.JSONDecodeError:
                pass 
                
        lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if osp.exists(cache_path):
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                TVRID_Depth_CombinedSplit._pid_to_samples_cache[cache_key] = data
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
                return data

            pid_to_samples = {}

            def _add_rows(rows, base_dir, pid_prefix, default_cam):
                import sys
                total = len(rows)
                for idx, row in enumerate(rows):
                    if idx % 500 == 0:
                        print(f"Caching single best depth frames for validation: {idx}/{total}...")
                        sys.stdout.flush()

                    person_id = row["person_id"]
                    cam_name = row.get("cam_name", "")
                    path = row["path"].replace("\\", "/")

                    folder_path = osp.join(base_dir, "train", path)
                    img_path = get_best_frame(folder_path, "depth", "largest")
                    if img_path is None:
                        continue

                    cam_id = self.CAM_NAME_TO_ID.get(cam_name, default_cam)
                    pid = f"{pid_prefix}{person_id}"
                    pid_to_samples.setdefault(pid, []).append((img_path, pid, cam_id))

            # DB_extracted
            db_rows = load_csv_data(self.db_train_csv)
            _add_rows(db_rows, self.db_dir, f"{self.dataset_name}_db_", 0)

            # TVPR_extracted (Original TVPR)
            if hasattr(self, 'tvpr1_train_csv') and osp.exists(self.tvpr1_train_csv):
                tvpr1_rows = load_csv_data(self.tvpr1_train_csv)
                _add_rows(tvpr1_rows, self.tvpr1_dir, f"{self.dataset_name}_tvpr1_", 5) # Distinguish from TVPR_2

            # TVPR_2_extracted
            if osp.exists(self.tvpr_train_csv):
                tvpr_rows = load_csv_data(self.tvpr_train_csv)
                _add_rows(tvpr_rows, self.tvpr_dir, f"{self.dataset_name}_tvpr2_", 4)

            # Deterministic order for reproducibility
            for pid in list(pid_to_samples.keys()):
                pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[0])

            # Save disk cache
            with open(cache_path, 'w') as f:
                json.dump(pid_to_samples, f)
            
            # Save memory cache
            TVRID_Depth_CombinedSplit._pid_to_samples_cache[cache_key] = pid_to_samples
            
        except Exception as e:
            print(f"Could not save single best depth frames cache: {e}")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

        return pid_to_samples

    def process_train(self):
        pid_to_samples = self._collect_depth_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        data = []
        for pid, samples in pid_to_samples.items():
            if pid in val_pids:
                continue
            for img_path, pid_str, cam_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((img_path, pid_str, camid))
        return data

    def process_test(self, mode='query'):
        pid_to_samples = self._collect_depth_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            samples = pid_to_samples.get(pid, [])
            if len(samples) < 2:
                continue

            # Choose a deterministic query index per identity
            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            q_sample = samples[qidx]
            q_img, q_pid, _ = q_sample
            query.append((q_img, q_pid, 0))

            for j, (img_path, pid_str, _cam_id) in enumerate(samples):
                if j == qidx:
                    continue
                gallery.append((img_path, pid_str, 1))

        return query if mode == 'query' else gallery


@DATASET_REGISTRY.register()
class TVRID_Depth_CombinedSplit_DBStratified(TVRID_Depth_CombinedSplit):
    """Like `TVRID_Depth_CombinedSplit`, but creates a stratified val split.

    Competition is DB-focused, so we bias the validation identities toward DB.
    We still keep a small TVPR portion to retain some out-of-domain signal.

    Split is identity-disjoint from training and deterministic.
    """

    dataset_name = "tvrid_depth_combined_dbfocus"

    # Per-domain identity ratios for validation
    VAL_ID_RATIO_DB = 0.35
    VAL_ID_RATIO_TVPR = 0.02
    SPLIT_SEED = 42

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            (
                f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}"
                f"_db{self.VAL_ID_RATIO_DB:.2f}_tvpr{self.VAL_ID_RATIO_TVPR:.2f}.json"
            ),
        )

    def _load_or_create_val_pids(self, pid_to_samples):
        split_path = self._split_file()
        if osp.exists(split_path):
            try:
                with open(split_path, "r") as f:
                    obj = json.load(f)
                val_pids = set(obj.get("val_pids", []))
                if val_pids:
                    return val_pids
            except Exception:
                pass

        candidates = sorted([pid for pid, samples in pid_to_samples.items() if len(samples) >= 2])
        if not candidates:
            return set()

        db_candidates = [pid for pid in candidates if "_db_" in pid]
        tvpr_candidates = [pid for pid in candidates if "_tvpr_" in pid]

        def _clamp_size(n, ratio):
            if n <= 0:
                return 0
            size = int(round(n * float(ratio)))
            if n == 1:
                return 1
            return max(1, min(size, n - 1))

        db_val_size = _clamp_size(len(db_candidates), self.VAL_ID_RATIO_DB)
        tvpr_val_size = _clamp_size(len(tvpr_candidates), self.VAL_ID_RATIO_TVPR)

        rng = hashlib.md5(f"{self.dataset_name}:{self.SPLIT_SEED}".encode("utf-8")).hexdigest()
        db_sorted = sorted(db_candidates, key=lambda pid: self._stable_int(pid + rng + ":db"))
        tvpr_sorted = sorted(tvpr_candidates, key=lambda pid: self._stable_int(pid + rng + ":tvpr"))

        val_pids = set(db_sorted[:db_val_size] + tvpr_sorted[:tvpr_val_size])

        payload = {
            "dataset": self.dataset_name,
            "seed": self.SPLIT_SEED,
            "val_id_ratio_db": self.VAL_ID_RATIO_DB,
            "val_id_ratio_tvpr": self.VAL_ID_RATIO_TVPR,
            "num_ids_total": len(pid_to_samples),
            "num_ids_candidates": len(candidates),
            "num_ids_db_candidates": len(db_candidates),
            "num_ids_tvpr_candidates": len(tvpr_candidates),
            "num_ids_val": len(val_pids),
            "num_ids_db_val": sum("_db_" in p for p in val_pids),
            "num_ids_tvpr_val": sum("_tvpr_" in p for p in val_pids),
            "val_pids": sorted(val_pids),
        }
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=osp.dirname(split_path)) as tf:
                json.dump(payload, tf, indent=2)
                tmp_name = tf.name
            os.replace(tmp_name, split_path)
        except Exception:
            pass

        return val_pids


@DATASET_REGISTRY.register()
class TVRID_Depth_CombinedSplit_DBHalf(TVRID_Depth_CombinedSplit):
    """TVRID Depth dataset with Combined sources but 50% DB reserved for validation.
    
    Same split strategy as TVRID_RGB_CombinedSplit_DBHalf for consistency.
    """

    dataset_name = "tvrid_depth_combined_dbhalf"

    # Per-domain identity ratios for validation
    # 50% of DB for validation -> Stronger DB evaluation
    # 0% of TVPR for validation -> Use ALL TVPR for training
    VAL_ID_RATIO_DB = 0.50
    VAL_ID_RATIO_TVPR = 0.0
    SPLIT_SEED = 42

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            (
                f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}"
                f"_db{self.VAL_ID_RATIO_DB:.2f}_tvpr{self.VAL_ID_RATIO_TVPR:.2f}.json"
            ),
        )

    def _load_or_create_val_pids(self, pid_to_samples):
        return TVRID_Depth_CombinedSplit_DBStratified._load_or_create_val_pids(self, pid_to_samples)


@DATASET_REGISTRY.register()
class TVRID_Depth_AllCombined(TVRID_Depth_CombinedSplit):
    """TVRID Depth dataset with ALL THREE sources (DB, TVPR, TVPR_2).
    
    Reserves ONLY 15% of DB_extracted for validation, maximizing the number of DB 
    identities available for training mapping to the competition target domain.
    0% of TVPR and TVPR2 are used for validation to act entirely as supplemental training data.
    """

    dataset_name = "tvrid_depth_all_combined"

    # Per-domain identity ratios for validation
    # 15% of DB for validation -> Leaves 85% DB for training
    # 0% of TVPR/TVPR2 for validation -> Use ALL TVPR/TVPR2 for training
    VAL_ID_RATIO_DB = 0.15
    VAL_ID_RATIO_TVPR = 0.0
    SPLIT_SEED = 42

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            (
                f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}"
                f"_db{self.VAL_ID_RATIO_DB:.2f}_tvpr{self.VAL_ID_RATIO_TVPR:.2f}.json"
            ),
        )

    def _load_or_create_val_pids(self, pid_to_samples):
        return TVRID_Depth_CombinedSplit_DBStratified._load_or_create_val_pids(self, pid_to_samples)


@DATASET_REGISTRY.register()
class TVRID_Depth_DBOnlySplit(TVRIDBase):
    """DB_extracted-only depth dataset with a leakage-free, identity-disjoint val split.

    Intended for a more reasonable *DB-only* validation proxy than `TVRID_Depth`.
    """

    dataset_name = "tvrid_depth_dbonly"

    VAL_ID_RATIO = 0.2
    SPLIT_SEED = 42

    def _stable_int(self, s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}_r{self.VAL_ID_RATIO:.2f}.json",
        )

    def _load_or_create_val_pids(self, pid_to_samples):
        split_path = self._split_file()
        if osp.exists(split_path):
            try:
                with open(split_path, "r") as f:
                    obj = json.load(f)
                val_pids = set(obj.get("val_pids", []))
                if val_pids:
                    return val_pids
            except Exception:
                pass

        candidates = sorted([pid for pid, samples in pid_to_samples.items() if len(samples) >= 2])
        if not candidates:
            return set()

        val_size = int(round(len(candidates) * float(self.VAL_ID_RATIO)))
        val_size = max(1, min(val_size, len(candidates) - 1)) if len(candidates) > 1 else 1

        rng = hashlib.md5(f"{self.dataset_name}:{self.SPLIT_SEED}".encode("utf-8")).hexdigest()
        salted = sorted(candidates, key=lambda pid: self._stable_int(pid + rng))
        val_pids = set(salted[:val_size])

        payload = {
            "dataset": self.dataset_name,
            "seed": self.SPLIT_SEED,
            "val_id_ratio": self.VAL_ID_RATIO,
            "num_ids_total": len(pid_to_samples),
            "num_ids_candidates": len(candidates),
            "num_ids_val": len(val_pids),
            "val_pids": sorted(val_pids),
        }
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=osp.dirname(split_path)) as tf:
                json.dump(payload, tf, indent=2)
                tmp_name = tf.name
            os.replace(tmp_name, split_path)
        except Exception:
            pass

        return val_pids

    def _collect_db_depth_samples(self):
        pid_to_samples = {}
        db_rows = load_csv_data(self.db_train_csv)
        for row in db_rows:
            person_id = row["person_id"]
            cam_name = row.get("cam_name", "")
            path = row["path"].replace("\\", "/")

            folder_path = osp.join(self.db_dir, "train", path)
            img_path = get_best_frame(folder_path, "depth", "largest")
            if img_path is None:
                continue

            cam_id = self.CAM_NAME_TO_ID.get(cam_name, 0)
            pid = f"{self.dataset_name}_{person_id}"
            pid_to_samples.setdefault(pid, []).append((img_path, pid, cam_id))

        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[0])
        return pid_to_samples

    def process_train(self):
        pid_to_samples = self._collect_db_depth_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        data = []
        for pid, samples in pid_to_samples.items():
            if pid in val_pids:
                continue
            for img_path, pid_str, cam_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((img_path, pid_str, camid))
        return data

    def process_test(self, mode='query'):
        pid_to_samples = self._collect_db_depth_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            samples = pid_to_samples.get(pid, [])
            if len(samples) < 2:
                continue

            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            q_img, q_pid, _ = samples[qidx]
            query.append((q_img, q_pid, 0))

            for j, (img_path, pid_str, _cam_id) in enumerate(samples):
                if j == qidx:
                    continue
                gallery.append((img_path, pid_str, 1))

        return query if mode == 'query' else gallery


@DATASET_REGISTRY.register()
class TVRID_Cross(TVRIDBase):
    """TVRID Cross-modal dataset for RGB<->Depth track.
    
    For training: combines RGB and Depth with modality-aware cam_ids.
    For testing: RGB as query, Depth as gallery.
    """
    
    dataset_name = "tvrid_cross"
    
    def process_train(self):
        """Process train with both RGB and Depth images.
        
        cam_id encoding: base_cam_id * 2 + modality (0=RGB, 1=Depth)
        """
        data = []
        
        # Process DB_extracted
        db_data = load_csv_data(self.db_train_csv)
        for row in db_data:
            person_id = row['person_id']
            cam_name = row['cam_name']
            path = row['path'].replace('\\', '/')
            
            base_cam_id = self.CAM_NAME_TO_ID.get(cam_name, 0)
            folder_path = osp.join(self.db_dir, 'train', path)
            pid = f"{self.dataset_name}_{person_id}"
            
            # Add RGB image
            rgb_path = get_middle_frame(folder_path, 'RGB')
            if rgb_path:
                rgb_camid = f"{self.dataset_name}_{base_cam_id * 2}"
                data.append((rgb_path, pid, rgb_camid))
            
            # Add Depth image
            depth_path = get_best_frame(folder_path, 'depth', 'largest')
            if depth_path:
                depth_camid = f"{self.dataset_name}_{base_cam_id * 2 + 1}"
                data.append((depth_path, pid, depth_camid))
        
        # Process TVPR_2_extracted if available
        if osp.exists(self.tvpr_train_csv):
            tvpr_data = load_csv_data(self.tvpr_train_csv)
            for row in tvpr_data:
                person_id = row['person_id']
                cam_name = row['cam_name']
                path = row['path'].replace('\\', '/')
                
                base_cam_id = self.CAM_NAME_TO_ID.get(cam_name, 4)
                folder_path = osp.join(self.tvpr_dir, 'train', path)
                pid = f"{self.dataset_name}_tvpr_{person_id}"
                
                # Add RGB image
                rgb_path = get_middle_frame(folder_path, 'RGB')
                if rgb_path:
                    rgb_camid = f"{self.dataset_name}_{base_cam_id * 2}"
                    data.append((rgb_path, pid, rgb_camid))
                
                # Add Depth image
                depth_path = get_best_frame(folder_path, 'depth', 'largest')
                if depth_path:
                    depth_camid = f"{self.dataset_name}_{base_cam_id * 2 + 1}"
                    data.append((depth_path, pid, depth_camid))
        
        return data
    
    def process_test(self, mode='query'):
        """For cross-modal test: RGB as query, Depth as gallery."""
        data = []
        
        test_data = load_csv_data(self.db_test_csv)
        modality = 'RGB' if mode == 'query' else 'depth'
        
        for row in test_data:
            gallery_id = row['gallery_id']
            path = row['path'].replace('\\', '/')
            
            folder_path = osp.join(self.db_dir, 'test_public', path)
            img_path = get_middle_frame(folder_path, modality)
            
            if img_path is None:
                continue
            
            # cam_id encodes modality for cross-modal
            camid = f"{self.dataset_name}_query" if mode == 'query' else f"{self.dataset_name}_gallery"
            data.append((img_path, gallery_id, camid))
        
        return data

@DATASET_REGISTRY.register()
class TVRID_Depth_TVPROnlySplit(TVRID_Depth_CombinedSplit):
    """TVPR_extracted-only depth dataset with a leakage-free, identity-disjoint val split.
    
    Intended for evaluation on TVPR-only subset.
    """

    dataset_name = "tvrid_depth_tvpronly"
    VAL_ID_RATIO = 0.2

    def _collect_depth_samples(self):
        """Collect depth samples from ONLY TVPR_2_extracted."""
        pid_to_samples = {}

        def _add_rows(rows, base_dir, pid_prefix, default_cam):
            for row in rows:
                person_id = row["person_id"]
                cam_name = row.get("cam_name", "")
                path = row["path"].replace("\\", "/")

                folder_path = osp.join(base_dir, "train", path)
                img_path = get_best_frame(folder_path, "depth", "largest")
                if img_path is None:
                    continue

                cam_id = self.CAM_NAME_TO_ID.get(cam_name, default_cam)
                pid = f"{pid_prefix}{person_id}"
                pid_to_samples.setdefault(pid, []).append((img_path, pid, cam_id))

        # TVPR_2_extracted
        if osp.exists(self.tvpr_train_csv):
            tvpr_rows = load_csv_data(self.tvpr_train_csv)
            _add_rows(tvpr_rows, self.tvpr_dir, f"{self.dataset_name}_tvpr_", 4)

        # Deterministic order for reproducibility
        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[0])

        return pid_to_samples

@DATASET_REGISTRY.register()
class TVRID_RGB_CombinedSplit(TVRID_Depth_CombinedSplit):
    """TVRID RGB-only dataset with combined sources and a leakage-free val split."""
    
    dataset_name = "tvrid_rgb_combined"

    def _collect_rgb_samples(self):
        """Collect RGB samples from both DB_extracted and TVPR_2_extracted."""
        pid_to_samples = {}

        def _add_rows(rows, base_dir, pid_prefix, default_cam):
            for row in rows:
                person_id = row["person_id"]
                cam_name = row.get("cam_name", "")
                path = row["path"].replace("\\", "/")

                folder_path = osp.join(base_dir, "train", path)
                img_path = get_middle_frame(folder_path, "RGB")
                if img_path is None:
                    continue

                cam_id = self.CAM_NAME_TO_ID.get(cam_name, default_cam)
                pid = f"{pid_prefix}{person_id}"
                pid_to_samples.setdefault(pid, []).append((img_path, pid, cam_id))

        # DB_extracted
        db_rows = load_csv_data(self.db_train_csv)
        _add_rows(db_rows, self.db_dir, f"{self.dataset_name}_db_", 0)

        # TVPR_2_extracted
        if osp.exists(self.tvpr_train_csv):
            tvpr_rows = load_csv_data(self.tvpr_train_csv)
            _add_rows(tvpr_rows, self.tvpr_dir, f"{self.dataset_name}_tvpr_", 4)

        # Deterministic order
        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[0])

        return pid_to_samples

    def process_train(self):
        # Override to call _collect_rgb_samples
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)
        
        data = []
        for pid, samples in pid_to_samples.items():
            if pid in val_pids:
                continue
            for img_path, pid_str, cam_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((img_path, pid_str, camid))
        return data

    def process_test(self, mode='query'):
        # Override to call _collect_rgb_samples
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            samples = pid_to_samples.get(pid, [])
            if len(samples) < 2:
                continue

            # Same logic as Depth: choose 1 query, rest gallery
            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            q_sample = samples[qidx]
            q_img, q_pid, q_camid = q_sample
            query.append((q_img, q_pid, q_camid))

            for j, (img_path, pid_str, g_camid) in enumerate(samples):
                if j == qidx:
                    continue
                gallery.append((img_path, pid_str, g_camid))

        return query if mode == 'query' else gallery

@DATASET_REGISTRY.register()
class TVRID_RGB_CombinedSplit_DBStratified(TVRID_RGB_CombinedSplit):
    """TVRID RGB-only dataset with Stratified DB-focused validation split."""

    dataset_name = "tvrid_rgb_combined_dbfocus"

    # Per-domain identity ratios for validation
    VAL_ID_RATIO_DB = 0.35
    VAL_ID_RATIO_TVPR = 0.02
    SPLIT_SEED = 42

    def _split_file(self) -> str:
        # Same as TVRID_Depth_CombinedSplit_DBStratified but uses self.dataset_name
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            (
                f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}"
                f"_db{self.VAL_ID_RATIO_DB:.2f}_tvpr{self.VAL_ID_RATIO_TVPR:.2f}.json"
            ),
        )

    def _load_or_create_val_pids(self, pid_to_samples):
        # Re-use logic from TVRID_Depth_CombinedSplit_DBStratified
        return TVRID_Depth_CombinedSplit_DBStratified._load_or_create_val_pids(self, pid_to_samples)
    
    def process_test(self, mode='query'):
        # Override to filter for DB-only evaluation
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            # STRICT FILTER: Only evaluate on DB identities
            if "_db_" not in pid:
                continue

            samples = pid_to_samples.get(pid, [])
            if len(samples) < 2:
                continue

            # Same logic as Depth: choose 1 query, rest gallery
            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            
            # Unpack 3 elements (img, pid, camid)
            q_sample = samples[qidx]
            q_img, q_pid, q_camid = q_sample
            query.append((q_img, q_pid, q_camid))

            for j, (img_path, pid_str, g_camid) in enumerate(samples):
                if j == qidx:
                    continue
                gallery.append((img_path, pid_str, g_camid))

        return query if mode == 'query' else gallery
@DATASET_REGISTRY.register()
class TVRID_RGB_DBOnlySplit(TVRID_RGB_CombinedSplit):
    """DB_extracted-only RGB dataset with a leakage-free, identity-disjoint val split.
    
    Intended for evaluation on DB-only subset.
    """

    dataset_name = "tvrid_rgb_dbonly"
    VAL_ID_RATIO = 0.2

    def _collect_rgb_samples(self):
        """Collect RGB samples from ONLY DB_extracted."""
        pid_to_samples = {}

        def _add_rows(rows, base_dir, pid_prefix, default_cam):
            for row in rows:
                person_id = row["person_id"]
                cam_name = row.get("cam_name", "")
                path = row["path"].replace("\\", "/")

                folder_path = osp.join(base_dir, "train", path)
                img_path = get_middle_frame(folder_path, "RGB")
                if img_path is None:
                    continue

                cam_id = self.CAM_NAME_TO_ID.get(cam_name, default_cam)
                pid = f"{pid_prefix}{person_id}"
                pid_to_samples.setdefault(pid, []).append((img_path, pid, cam_id))

        # DB_extracted
        db_rows = load_csv_data(self.db_train_csv)
        _add_rows(db_rows, self.db_dir, f"{self.dataset_name}_db_", 0)

        # Deterministic order for reproducibility
        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[0])

        return pid_to_samples

@DATASET_REGISTRY.register()
class TVRID_RGB_TVPROnlySplit_MultiFrame(TVRID_RGB_DBOnlySplit):
    """TVPR_extracted-only RGB benchmark using multiple frames per passage."""

    dataset_name = "tvrid_rgb_tvpronly_multiframe"
    VAL_ID_RATIO = 0.2
    N_FRAMES = 20
    FRAME_STRATEGY = "middle_expand"

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}_r{self.VAL_ID_RATIO:.2f}.json",
        )

    def _collect_rgb_samples(self):
        """Collect one item per TVPR passage, each containing multiple RGB frames."""
        pid_to_samples = {}

        tvpr_rows = load_csv_data(self.tvpr1_train_csv)
        for row in tvpr_rows:
            person_id = row["person_id"]
            cam_name = row.get("cam_name", "")
            path = row["path"].replace("\\", "/")

            folder_path = osp.join(self.tvpr1_dir, "train", path)
            frame_paths = get_multiple_frames(
                folder_path, "RGB", self.N_FRAMES, self.FRAME_STRATEGY
            )
            if not frame_paths:
                continue

            cam_id = self.CAM_NAME_TO_ID.get(cam_name, 5)
            pid = f"tvrid_rgb_tvpronly_tvpr1_{person_id}"
            passage_id = f"{pid}:{path}"
            pid_to_samples.setdefault(pid, []).append((frame_paths, pid, cam_id, passage_id))

        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[3])

        return pid_to_samples

    def process_train(self):
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        data = []
        for pid, samples in pid_to_samples.items():
            if pid in val_pids:
                continue
            for frame_paths, pid_str, cam_id, passage_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((frame_paths, pid_str, camid, passage_id))
        return data

    def process_test(self, mode='query'):
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            samples = pid_to_samples.get(pid, [])
            if len(samples) < 2:
                continue

            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            q_frames, q_pid, _q_camid, q_passage_id = samples[qidx]
            for frame_path in q_frames:
                # TVPR is single-camera. Use split-specific camids so same-ID matches
                # are not filtered out as "same-camera" by Market-style evaluation.
                query.append((frame_path, q_pid, f"{self.dataset_name}_query", q_passage_id))

            for j, (frame_paths, pid_str, _g_camid, passage_id) in enumerate(samples):
                if j == qidx:
                    continue
                for frame_path in frame_paths:
                    gallery.append((frame_path, pid_str, f"{self.dataset_name}_gallery", passage_id))

        return query if mode == 'query' else gallery

@DATASET_REGISTRY.register()
class TVRID_RGB_TVPR2OnlySplit_MultiFrame(TVRID_RGB_DBOnlySplit):
    """TVPR_2_extracted-only RGB benchmark using multiple frames per passage."""

    dataset_name = "tvrid_rgb_tvpr2only_multiframe"
    VAL_ID_RATIO = 0.2
    N_FRAMES = 100
    FRAME_STRATEGY = "top_n"

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}_r{self.VAL_ID_RATIO:.2f}.json",
        )

    def _collect_rgb_samples(self):
        """Collect one item per TVPR2 passage, each containing multiple RGB frames."""
        pid_to_samples = {}

        tvpr_rows = load_csv_data(self.tvpr_train_csv)
        for row in tvpr_rows:
            person_id = row["person_id"]
            cam_name = row.get("cam_name", "")
            path = row["path"].replace("\\", "/")

            folder_path = osp.join(self.tvpr_dir, "train", path)
            frame_paths = get_multiple_frames(
                folder_path, "RGB", self.N_FRAMES, self.FRAME_STRATEGY
            )
            if not frame_paths:
                continue

            cam_id = self.CAM_NAME_TO_ID.get(cam_name, 4)
            pid = f"tvrid_rgb_tvpr2only_tvpr2_{person_id}"
            passage_id = f"{pid}:{path}"
            pid_to_samples.setdefault(pid, []).append((frame_paths, pid, cam_id, passage_id))

        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[3])

        return pid_to_samples

    def process_train(self):
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        data = []
        for pid, samples in pid_to_samples.items():
            if pid in val_pids:
                continue
            for frame_paths, pid_str, cam_id, passage_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((frame_paths, pid_str, camid, passage_id))
        return data

    def process_test(self, mode='query'):
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            samples = pid_to_samples.get(pid, [])
            if len(samples) < 2:
                continue

            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            q_frames, q_pid, _q_camid, q_passage_id = samples[qidx]
            for frame_path in q_frames:
                query.append((frame_path, q_pid, f"{self.dataset_name}_query", q_passage_id))

            for j, (frame_paths, pid_str, _g_camid, passage_id) in enumerate(samples):
                if j == qidx:
                    continue
                for frame_path in frame_paths:
                    gallery.append((frame_path, pid_str, f"{self.dataset_name}_gallery", passage_id))

        return query if mode == 'query' else gallery

@DATASET_REGISTRY.register()
class TVRID_RGB_CombinedSplit_DBHalf(TVRID_RGB_CombinedSplit):
    """TVRID RGB-only dataset with Combined sources but 50% DB reserved for validation."""

    dataset_name = "tvrid_rgb_combined_dbhalf"

    # Per-domain identity ratios for validation
    # 50% of DB for validation -> Stronger DB evaluation
    # 0% of TVPR for validation -> Use ALL TVPR for training to maximize source domain transfer
    VAL_ID_RATIO_DB = 0.50
    VAL_ID_RATIO_TVPR = 0.0
    SPLIT_SEED = 42

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            (
                f"{self.dataset_name}_valsplit_seed{self.SPLIT_SEED}"
                f"_db{self.VAL_ID_RATIO_DB:.2f}_tvpr{self.VAL_ID_RATIO_TVPR:.2f}.json"
            ),
        )

    def _load_or_create_val_pids(self, pid_to_samples):
        # Explicitly recycle the logic involving 2 ratios
        return TVRID_Depth_CombinedSplit_DBStratified._load_or_create_val_pids(self, pid_to_samples)

@DATASET_REGISTRY.register()
class TVRID_RGB_DBOnlySplit_MultiFrame(TVRID_RGB_DBOnlySplit):
    """DB-only RGB benchmark using multiple frames per passage.

    Training keeps one dataset item per passage and randomly samples a frame in
    CommDataset. Evaluation expands frames, then ReidEvaluator averages features
    back to one descriptor per passage.
    """

    dataset_name = "tvrid_rgb_dbonly_multiframe"
    N_FRAMES = 5
    FRAME_STRATEGY = "uniform"

    def _split_file(self) -> str:
        split_dir = osp.join(self.root, "data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        return osp.join(
            split_dir,
            f"tvrid_rgb_dbonly_valsplit_seed{self.SPLIT_SEED}_r{self.VAL_ID_RATIO:.2f}.json",
        )

    def _collect_rgb_samples(self):
        """Collect one item per DB passage, each containing multiple RGB frames."""
        pid_to_samples = {}

        db_rows = load_csv_data(self.db_train_csv)
        for row in db_rows:
            person_id = row["person_id"]
            cam_name = row.get("cam_name", "")
            path = row["path"].replace("\\", "/")

            folder_path = osp.join(self.db_dir, "train", path)
            frame_paths = get_multiple_frames(
                folder_path, "RGB", self.N_FRAMES, self.FRAME_STRATEGY
            )
            if not frame_paths:
                continue

            cam_id = self.CAM_NAME_TO_ID.get(cam_name, 0)
            pid = f"tvrid_rgb_dbonly_db_{person_id}"
            passage_id = f"{pid}:{path}"
            pid_to_samples.setdefault(pid, []).append((frame_paths, pid, cam_id, passage_id))

        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[3])

        return pid_to_samples

    def process_train(self):
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        data = []
        for pid, samples in pid_to_samples.items():
            if pid in val_pids:
                continue
            for frame_paths, pid_str, cam_id, passage_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((frame_paths, pid_str, camid, passage_id))
        return data

    def process_test(self, mode='query'):
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            samples = pid_to_samples.get(pid, [])
            if len(samples) < 2:
                continue

            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            q_frames, q_pid, q_camid, q_passage_id = samples[qidx]
            for frame_path in q_frames:
                query.append((frame_path, q_pid, f"{self.dataset_name}_{q_camid}", q_passage_id))

            for j, (frame_paths, pid_str, g_camid, passage_id) in enumerate(samples):
                if j == qidx:
                    continue
                for frame_path in frame_paths:
                    gallery.append((frame_path, pid_str, f"{self.dataset_name}_{g_camid}", passage_id))

        return query if mode == 'query' else gallery


@DATASET_REGISTRY.register()
class TVRID_RGB_DBPublic_MultiFrame(TVRID_RGB_DBOnlySplit_MultiFrame):
    """DB-only RGB benchmark using all DB train labels and public-test labels.

    Training uses every labelled DB training passage. Testing uses
    `pubic_test_ground_truth/test_secret_map.csv` for identity labels; each
    public-test passage is inserted into both query and gallery, with a unique
    camid so Market-style evaluation removes only the exact self-passage while
    keeping same-identity cross-passage matches.
    """

    dataset_name = "tvrid_rgb_dbpublic_multiframe"
    N_FRAMES = 90
    FRAME_STRATEGY = "top_n"

    def process_train(self):
        pid_to_samples = self._collect_rgb_samples()

        data = []
        for _pid, samples in pid_to_samples.items():
            for frame_paths, pid_str, cam_id, passage_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((frame_paths, pid_str, camid, passage_id))
        return data

    def process_test(self, mode='query'):
        self.check_before_run(self.db_public_secret_map_csv)
        test_rows = load_csv_data(self.db_public_secret_map_csv)
        data = []

        for i, row in enumerate(test_rows):
            public_gallery_id = row["public_gallery_id"]
            person_id = row["person_id"]
            folder_path = osp.join(self.db_dir, "test_public", public_gallery_id)
            frame_paths = get_multiple_frames(
                folder_path, "RGB", self.N_FRAMES, self.FRAME_STRATEGY
            )
            if not frame_paths:
                continue

            pid = f"{self.dataset_name}_public_{person_id}"
            camid = f"{self.dataset_name}_public_{i}"
            passage_id = f"{pid}:{public_gallery_id}"

            for frame_path in frame_paths:
                data.append((frame_path, pid, camid, passage_id))

        return data


@DATASET_REGISTRY.register()
class TVRID_RGB_DB_AllVal(TVRID_RGB_DBOnlySplit):
    """TVRID RGB dataset that uses ALL DB identities for validation/testing.
    
    WARNING: If used with a model trained on DB data, this includes training data 
    in the test set (Leakage), but provides the maximum number of identities for 
    checking model behavior on the target domain.
    """
    
    dataset_name = "tvrid_rgb_db_allval"
    
    def _load_or_create_val_pids(self, pid_to_samples):
        # Return ALL pids available in the collected samples
        # This ignores any split file and uses everything for testing
        return sorted(list(pid_to_samples.keys()))

@DATASET_REGISTRY.register()
class TVRID_RGB_CrossCameraVal(TVRID_RGB_DBOnlySplit):
    """TVRID RGB dataset with Cross-Camera validation to simulate public test scenarios.
    
    Public test scenarios:
    - same_cam_cross_passage: Same camera, different passage (handled by temporal difference)
    - up_down_cross_passage: Upward <-> Downward camera matching
    - flat_vs_others: Flat camera <-> Other viewpoints
    
    This validation split creates HARD cross-camera query/gallery pairs:
    - Query: Images from cameras {flat, upward}
    - Gallery: Images from cameras {downward, upsideDown}
    
    This forces the model to match across very different viewpoints.
    """
    
    dataset_name = "tvrid_rgb_crosscam"
    
    # Query cameras vs Gallery cameras
    QUERY_CAMS = {'flat', 'upward'}
    GALLERY_CAMS = {'downward', 'upsideDown'}
    
    VAL_ID_RATIO = 0.3  # Use 30% of IDs for validation
    
    def _collect_rgb_samples_by_camera(self):
        """Collect RGB samples organized by camera type."""
        pid_cam_to_samples = {}  # {(pid, cam_name): [(img_path, pid, cam_id), ...]}
        
        db_rows = load_csv_data(self.db_train_csv)
        for row in db_rows:
            person_id = row["person_id"]
            cam_name = row.get("cam_name", "")
            path = row["path"].replace("\\", "/")
            
            folder_path = osp.join(self.db_dir, "train", path)
            img_path = get_middle_frame(folder_path, "RGB")
            if img_path is None:
                continue
            
            cam_id = self.CAM_NAME_TO_ID.get(cam_name, 0)
            pid = f"{self.dataset_name}_db_{person_id}"
            
            key = (pid, cam_name)
            if key not in pid_cam_to_samples:
                pid_cam_to_samples[key] = []
            pid_cam_to_samples[key].append((img_path, pid, cam_id))
        
        return pid_cam_to_samples
    
    def process_train(self):
        """Training: Use all cameras from training IDs."""
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)
        
        data = []
        for pid, samples in pid_to_samples.items():
            if pid in val_pids:
                continue
            for img_path, pid_str, cam_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((img_path, pid_str, camid))
        return data
    
    def process_test(self, mode='query'):
        """Validation: Cross-camera protocol.
        
        Query: flat + upward cameras
        Gallery: downward + upsideDown cameras
        """
        pid_cam_to_samples = self._collect_rgb_samples_by_camera()
        pid_to_samples = self._collect_rgb_samples()
        val_pids = self._load_or_create_val_pids(pid_to_samples)
        
        query = []
        gallery = []
        
        for pid in sorted(val_pids):
            # Collect query images (flat + upward)
            for cam_name in self.QUERY_CAMS:
                key = (pid, cam_name)
                if key in pid_cam_to_samples:
                    for img_path, pid_str, cam_id in pid_cam_to_samples[key]:
                        # Use unique camid to avoid self-filtering
                        query.append((img_path, pid_str, f"query_{cam_id}"))
            
            # Collect gallery images (downward + upsideDown)
            for cam_name in self.GALLERY_CAMS:
                key = (pid, cam_name)
                if key in pid_cam_to_samples:
                    for img_path, pid_str, cam_id in pid_cam_to_samples[key]:
                        gallery.append((img_path, pid_str, f"gallery_{cam_id}"))
        
        return query if mode == 'query' else gallery


@DATASET_REGISTRY.register()
class TVRID_RGB_UpDownVal(TVRID_RGB_CrossCameraVal):
    """Specifically tests Upward <-> Downward matching (hardest scenario)."""
    
    dataset_name = "tvrid_rgb_updown"
    
    QUERY_CAMS = {'upward'}
    GALLERY_CAMS = {'downward'}


@DATASET_REGISTRY.register()
class TVRID_RGB_FlatVsOthersVal(TVRID_RGB_CrossCameraVal):
    """Tests Flat camera vs all other viewpoints."""
    
    dataset_name = "tvrid_rgb_flatvsothers"
    
    QUERY_CAMS = {'flat'}
    GALLERY_CAMS = {'upward', 'downward', 'upsideDown'}


@DATASET_REGISTRY.register()
class TVRID_Depth_CombinedSplit_DBHalf_MultiFrame(TVRID_Depth_CombinedSplit_DBHalf):
    """TVRID Depth dataset with MULTIPLE frames per passage for training.
    
    Instead of 1 best frame per passage (~2207 samples), uses N frames per passage.
    With N=5, yields ~10K+ training samples for much better metric learning.
    
    Validation STILL uses 1 best frame per identity for consistency.
    """

    dataset_name = "tvrid_depth_combined_dbhalf_multiframe"
    
    # Number of frames to use per passage during training
    N_FRAMES = 5
    FRAME_STRATEGY = 'top_n'  # 'top_n' (best depth), 'uniform', or 'all'
    
    # Class-level cache for expensive multi-frame collection
    _multiframe_cache = {}

    def _collect_depth_samples_multiframe(self):
        """Collect MULTIPLE depth frames per passage from DB and TVPR."""
        cache_key = f"{self.dataset_name}_{self.N_FRAMES}_{self.FRAME_STRATEGY}"
        if cache_key in TVRID_Depth_CombinedSplit_DBHalf_MultiFrame._multiframe_cache:
            return TVRID_Depth_CombinedSplit_DBHalf_MultiFrame._multiframe_cache[cache_key]
        
        pid_to_samples = {}

        def _add_rows(rows, base_dir, pid_prefix, default_cam):
            for row in rows:
                person_id = row["person_id"]
                cam_name = row.get("cam_name", "")
                path = row["path"].replace("\\", "/")

                folder_path = osp.join(base_dir, "train", path)
                frame_paths = get_multiple_frames(
                    folder_path, "depth", self.N_FRAMES, self.FRAME_STRATEGY
                )
                if not frame_paths:
                    continue

                cam_id = self.CAM_NAME_TO_ID.get(cam_name, default_cam)
                pid = f"{pid_prefix}{person_id}"
                for img_path in frame_paths:
                    pid_to_samples.setdefault(pid, []).append((img_path, pid, cam_id))

        # DB_extracted
        db_rows = load_csv_data(self.db_train_csv)
        _add_rows(db_rows, self.db_dir, f"{self.dataset_name}_db_", 0)

        # TVPR_2_extracted
        if osp.exists(self.tvpr_train_csv):
            tvpr_rows = load_csv_data(self.tvpr_train_csv)
            _add_rows(tvpr_rows, self.tvpr_dir, f"{self.dataset_name}_tvpr_", 4)

        # Deterministic order
        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[0])

        TVRID_Depth_CombinedSplit_DBHalf_MultiFrame._multiframe_cache[cache_key] = pid_to_samples
        return pid_to_samples

    def process_train(self):
        """Training uses multiple frames per passage."""
        pid_to_samples = self._collect_depth_samples_multiframe()
        # Re-use parent's val split logic but with the PARENT's single-frame data
        # to keep the same val identity set
        single_frame_data = self._collect_depth_samples()
        val_pids = self._load_or_create_val_pids(single_frame_data)

        data = []
        for pid, samples in pid_to_samples.items():
            # Map pid to parent dataset's pid format for val check
            parent_pid = pid.replace(self.dataset_name, "tvrid_depth_combined_dbhalf")
            if parent_pid in val_pids:
                continue
            for img_path, pid_str, cam_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((img_path, pid_str, camid))
        return data

    def process_test(self, mode='query'):
        """Validation still uses single best frame (same as parent)."""
        # Use parent's single-frame data for validation consistency
        single_frame_data = self._collect_depth_samples()
        val_pids = self._load_or_create_val_pids(single_frame_data)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            samples = single_frame_data.get(pid, [])
            if len(samples) < 2:
                continue

            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            q_sample = samples[qidx]
            q_img, q_pid, _ = q_sample
            query.append((q_img, q_pid, 0))

            for j, (img_path, pid_str, _cam_id) in enumerate(samples):
                if j == qidx:
                    continue
                gallery.append((img_path, pid_str, 1))

        return query if mode == 'query' else gallery


@DATASET_REGISTRY.register()
class TVRID_Depth_Dynamic_Combined(TVRID_Depth_AllCombined):
    """TVRID Depth dataset with DYNAMIC randomly sampled CORE frames for training.
    
    Instead of passing a static middle frame to the Datatset loader, this passes 
    a LIST of the top 50% most informative frames (Core frames). At every training 
    epoch, the loader randomly picks 1 frame from this list.
    
    Validation STILL uses 1 best frame per identity for consistency.
    """

    dataset_name = "tvrid_depth_dynamic_combined"
    
    def _collect_depth_samples_dynamic(self):
        """Collect lists of core depth frames per passage, with JSON caching for speed."""
        import fcntl
        import time
        import os

        # Use an absolute stable path for cache
        cache_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, f"{self.dataset_name}_core_frames_cache.json")
        lock_path = cache_path + ".lock"

        # Check if cache already exists before trying to acquire lock
        if osp.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass # Cache might be corrupted or still writing, proceed to lock

        # Acquire lock to ensure only one worker processes the cache
        lock_file = open(lock_path, 'w')
        try:
            # Exclusive lock, but wait until it's released if another worker has it
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            
            # Check AGAIN inside the lock, because another worker might have finished building it while we waited
            if osp.exists(cache_path):
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
                return data

            pid_to_samples = {}

            def _add_rows(rows, base_dir, pid_prefix, default_cam):
                import sys
                total = len(rows)
                for idx, row in enumerate(rows):
                    if idx % 500 == 0:
                        print(f"Caching dynamic core frames: {idx}/{total}...")
                        sys.stdout.flush()

                    person_id = row["person_id"]
                    cam_name = row.get("cam_name", "")
                    path = row["path"].replace("\\", "/")

                    folder_path = osp.join(base_dir, "train", path)
                    # Feed the top 50% largest depth maps into the list
                    frame_paths = get_core_frames(folder_path, "depth", ratio=0.5)
                    
                    if not frame_paths:
                        continue

                    cam_id = self.CAM_NAME_TO_ID.get(cam_name, default_cam)
                    pid = f"{pid_prefix}{person_id}"
                    
                    # IMPORTANT: We only append ONE tuple per passage, and pass the ENTIRE LIST as the "img_path"
                    pid_to_samples.setdefault(pid, []).append((frame_paths, pid, cam_id))

            # DB_extracted
            db_rows = load_csv_data(self.db_train_csv)
            _add_rows(db_rows, self.db_dir, f"{self.dataset_name}_db_", 0)

            # TVPR_extracted (Original TVPR)
            if hasattr(self, 'tvpr1_train_csv') and osp.exists(self.tvpr1_train_csv):
                tvpr1_rows = load_csv_data(self.tvpr1_train_csv)
                _add_rows(tvpr1_rows, self.tvpr1_dir, f"{self.dataset_name}_tvpr1_", 5)

            # TVPR_2_extracted
            if osp.exists(self.tvpr_train_csv):
                tvpr_rows = load_csv_data(self.tvpr_train_csv)
                _add_rows(tvpr_rows, self.tvpr_dir, f"{self.dataset_name}_tvpr2_", 4)

            # Deterministic order
            for pid in list(pid_to_samples.keys()):
                # Fall back to string comparison on the first path in the list
                pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[0][0])

            # Save cache
            with open(cache_path, 'w') as f:
                json.dump(pid_to_samples, f)
                
        except Exception as e:
            print(f"Could not save cache to {cache_path}: {e}")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

        return pid_to_samples

    # Number of times to duplicate DB identities to rebalance against TVPR dominance
    DB_OVERSAMPLE = 4

    def process_train(self):
        """Training uses dynamically sampled core frames per passage.
        
        DB identities are oversampled by duplicating them under alias PIDs
        so that NaiveIdentitySampler picks them more frequently.
        """
        pid_to_samples = self._collect_depth_samples_dynamic()
        # Ensure validation split strictly uses single best frame logic for deterministic hashes
        single_frame_data = self._collect_depth_samples()
        val_pids = self._load_or_create_val_pids(single_frame_data)

        data = []
        for pid, samples in pid_to_samples.items():
            # Map pid to parent dataset's pid format to check against val_pids
            parent_pid = pid.replace(self.dataset_name, "tvrid_depth_all_combined")
            if parent_pid in val_pids:
                continue

            # Detect if this is a DB identity (prefix contains "_db_")
            is_db = "_db_" in pid

            for img_paths, pid_str, cam_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                # data item `img_paths` is a LIST of paths !
                data.append((img_paths, pid_str, camid))

                # Oversample DB identities by creating alias PIDs
                if is_db:
                    for dup_idx in range(1, self.DB_OVERSAMPLE):
                        alias_pid = f"{pid_str}_dup{dup_idx}"
                        data.append((img_paths, alias_pid, camid))
        return data


@DATASET_REGISTRY.register()
class TVRID_Depth_AllCombined_MultiFrame(TVRID_Depth_AllCombined):
    """TVRID Depth dataset with ALL THREE frames and MULTIPLE frames per passage for training.
    
    Validation STILL uses 1 best frame per identity for consistency.
    """

    dataset_name = "tvrid_depth_all_combined_multiframe"
    
    # Number of frames to use per passage during training
    N_FRAMES = 5
    FRAME_STRATEGY = 'top_n'  # 'top_n' (best depth), 'uniform', or 'all'
    
    # Class-level cache for expensive multi-frame collection
    _multiframe_cache = {}

    def _collect_depth_samples_multiframe(self):
        """Collect MULTIPLE depth frames per passage from DB and TVPR."""
        cache_key = f"{self.dataset_name}_{self.N_FRAMES}_{self.FRAME_STRATEGY}"
        if cache_key in TVRID_Depth_AllCombined_MultiFrame._multiframe_cache:
            return TVRID_Depth_AllCombined_MultiFrame._multiframe_cache[cache_key]
        
        pid_to_samples = {}

        def _add_rows(rows, base_dir, pid_prefix, default_cam):
            for row in rows:
                person_id = row["person_id"]
                cam_name = row.get("cam_name", "")
                path = row["path"].replace("\\", "/")

                folder_path = osp.join(base_dir, "train", path)
                frame_paths = get_multiple_frames(
                    folder_path, "depth", self.N_FRAMES, self.FRAME_STRATEGY
                )
                if not frame_paths:
                    continue

                cam_id = self.CAM_NAME_TO_ID.get(cam_name, default_cam)
                pid = f"{pid_prefix}{person_id}"
                for img_path in frame_paths:
                    pid_to_samples.setdefault(pid, []).append((img_path, pid, cam_id))

        # DB_extracted
        db_rows = load_csv_data(self.db_train_csv)
        _add_rows(db_rows, self.db_dir, f"{self.dataset_name}_db_", 0)

        # TVPR_extracted (Original TVPR)
        if hasattr(self, 'tvpr1_train_csv') and osp.exists(self.tvpr1_train_csv):
            tvpr1_rows = load_csv_data(self.tvpr1_train_csv)
            _add_rows(tvpr1_rows, self.tvpr1_dir, f"{self.dataset_name}_tvpr1_", 5)

        # TVPR_2_extracted
        if osp.exists(self.tvpr_train_csv):
            tvpr_rows = load_csv_data(self.tvpr_train_csv)
            _add_rows(tvpr_rows, self.tvpr_dir, f"{self.dataset_name}_tvpr2_", 4)

        # Deterministic order
        for pid in list(pid_to_samples.keys()):
            pid_to_samples[pid] = sorted(pid_to_samples[pid], key=lambda t: t[0])

        TVRID_Depth_AllCombined_MultiFrame._multiframe_cache[cache_key] = pid_to_samples
        return pid_to_samples

    def process_train(self):
        """Training uses multiple frames per passage."""
        pid_to_samples = self._collect_depth_samples_multiframe()
        single_frame_data = self._collect_depth_samples()
        val_pids = self._load_or_create_val_pids(single_frame_data)

        data = []
        for pid, samples in pid_to_samples.items():
            if pid in val_pids:
                continue
            for img_path, pid_str, cam_id in samples:
                camid = f"{self.dataset_name}_{cam_id}"
                data.append((img_path, pid_str, camid))
        return data

    def process_test(self, mode='query'):
        """Validation still uses single best frame (same as parent)."""
        single_frame_data = self._collect_depth_samples()
        val_pids = self._load_or_create_val_pids(single_frame_data)

        query = []
        gallery = []

        for pid in sorted(val_pids):
            samples = single_frame_data.get(pid, [])
            if len(samples) < 2:
                continue

            qidx = (self._stable_int(pid) + int(self.SPLIT_SEED)) % len(samples)
            q_sample = samples[qidx]
            q_img, q_pid, _ = q_sample
            query.append((q_img, q_pid, 0))

            for j, (img_path, pid_str, _cam_id) in enumerate(samples):
                if j == qidx:
                    continue
                gallery.append((img_path, pid_str, 1))

        return query if mode == 'query' else gallery
