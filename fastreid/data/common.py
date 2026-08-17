# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import (
    read_image, read_image_depth_masked,
    load_precomputed_mask, apply_body_part_erasing, apply_background_erasing,
    apply_background_alternating,
)


class CommDataset(Dataset):
    """Image Person ReID Dataset
    
    Supports optional mask-based augmentations controlled via augment_cfg dict:
        bpe: {enabled, prob, erase_max}                       Body Part Erasing (REA on body)
        bge: {enabled, prob}                                   Background Erasing
    These are applied BEFORE standard transforms on the original-resolution image.
    """

    def __init__(self, img_items, transform=None, relabel=True, augment_cfg=None):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.augment_cfg = augment_cfg or {}

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        def _numeric_key(x):
            # Sort "dataset_N" strings by N as integer; fall back to lexicographic
            try:
                return int(str(x).rsplit('_', 1)[-1])
            except ValueError:
                return x

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set), key=_numeric_key)
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def _apply_mask_augmentations(self, img, img_path):
        """Apply BPE / BGE if enabled — requires precomputed person masks.
        
        Each augmentation fires independently with its own probability,
        consistent with how REA, LGPR, GGPR etc. work in the pipeline.
        """
        bpe = self.augment_cfg.get('bpe')
        bge = self.augment_cfg.get('bge')
        bga = self.augment_cfg.get('bga')

        if not bpe and not bge and not bga:
            return img

        mask = load_precomputed_mask(img_path)
        if mask is None:
            return img

        if bpe:
            img = apply_body_part_erasing(
                img, mask,
                erase_prob=bpe.get('prob', 0.5),
                erase_max=bpe.get('erase_max', 3),
            )
        if bge:
            img = apply_background_erasing(
                img, mask,
                erase_prob=bge.get('prob', 0.5),
            )
            
        if bga:
            img = apply_background_alternating(
                img, mask,
                prob=bga.get('prob', 0.5),
                contrast_thresh=bga.get('contrast_thresh', 80.0),
                mode=bga.get('mode', 'random_color'),
                specific_color=bga.get('specific_color', (0, 0, 0)),
                noise_std=bga.get('noise_std', 30.0),
            )
            
        return img

    def __getitem__(self, index):
        import random
        img_item = self.img_items[index]
        img_path = img_item[0]
        
        # Dynamic Frame Sampling: If img_path is a list, pick a random frame
        if isinstance(img_path, list) and len(img_path) > 0:
            img_path = random.choice(img_path)
            
        pid = img_item[1]
        camid = img_item[2]
        passage_id = img_item[3] if len(img_item) > 3 else None
        img = read_image(img_path)
        img = self._apply_mask_augmentations(img, img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        item = {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }
        if passage_id is not None:
            item["passage_ids"] = passage_id
        return item

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class DepthMaskedCommDataset(CommDataset):
    """Like CommDataset but applies depth-based foreground masking to RGB images.
    
    Derives the depth path from the RGB path (<timestamp>_RGB.png → _depth.png),
    segments the person via adaptive depth thresholding, and fills the background
    with the training pixel mean so the model only sees person pixels.
    """

    def __getitem__(self, index):
        import random
        img_item = self.img_items[index]
        img_path = img_item[0]
        
        # Dynamic Frame Sampling: If img_path is a list, pick a random frame
        if isinstance(img_path, list) and len(img_path) > 0:
            img_path = random.choice(img_path)
            
        pid = img_item[1]
        camid = img_item[2]
        passage_id = img_item[3] if len(img_item) > 3 else None
        img = read_image_depth_masked(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        item = {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }
        if passage_id is not None:
            item["passage_ids"] = passage_id
        return item
