# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment


def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # crop
        do_crop = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE
        crop_scale = cfg.INPUT.CROP.SCALE
        crop_ratio = cfg.INPUT.CROP.RATIO

        # augmix augmentation
        do_augmix = cfg.INPUT.AUGMIX.ENABLED
        augmix_prob = cfg.INPUT.AUGMIX.PROB

        # auto augmentation
        do_autoaug = cfg.INPUT.AUTOAUG.ENABLED
        autoaug_prob = cfg.INPUT.AUTOAUG.PROB

        # horizontal filp
        do_flip = cfg.INPUT.FLIP.ENABLED
        flip_prob = cfg.INPUT.FLIP.PROB

        # padding
        do_pad = cfg.INPUT.PADDING.ENABLED
        padding_size = cfg.INPUT.PADDING.SIZE
        padding_mode = cfg.INPUT.PADDING.MODE

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE

        # random affine
        do_affine = cfg.INPUT.AFFINE.ENABLED

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_value = cfg.INPUT.REA.VALUE

        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        # local grayscale patch replacement
        do_lgpr = cfg.INPUT.LGPR.ENABLED
        lgpr_prob = cfg.INPUT.LGPR.PROB
        lgpr_patch_ratio = cfg.INPUT.LGPR.PATCH_RATIO

        # global grayscale patch replacement
        do_ggpr = cfg.INPUT.GGPR.ENABLED
        ggpr_prob = cfg.INPUT.GGPR.PROB

        if do_autoaug:
            res.append(T.RandomApply([AutoAugment()], p=autoaug_prob))

        if size_train[0] > 0:
            res.append(T.Resize(size_train[0] if len(size_train) == 1 else size_train, interpolation=3))

        if do_crop:
            res.append(T.RandomResizedCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size,
                                           interpolation=3,
                                           scale=crop_scale, ratio=crop_ratio))
        if do_pad:
            res.extend([T.Pad(padding_size, padding_mode=padding_mode),
                        T.RandomCrop(size_train[0] if len(size_train) == 1 else size_train)])
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
            # Vertical flip: simulates DB `upsideDown` camera angle
            do_vflip = getattr(cfg.INPUT.FLIP, "VERTICAL", False)
            if do_vflip:
                res.append(T.RandomVerticalFlip(p=0.3))

        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_lgpr:
            res.append(LGPR(prob=lgpr_prob, patch_ratio=lgpr_patch_ratio))
        if do_ggpr:
            res.append(GGPR(prob=ggpr_prob))
        if do_affine:
            affine_degrees = getattr(cfg.INPUT.AFFINE, "DEGREES", 45)
            perspective_scale = getattr(cfg.INPUT.AFFINE, "PERSPECTIVE_SCALE", 0.6)
            res.append(T.RandomPerspective(distortion_scale=perspective_scale, p=0.5))
            res.append(T.RandomAffine(degrees=affine_degrees, translate=None, scale=[0.9, 1.1], shear=0.1))
        if do_augmix:
            res.append(AugMix(prob=augmix_prob))
        res.append(ToTensor())
        if do_rea:
            res.append(T.RandomErasing(p=rea_prob, value=rea_value))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        do_crop = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE

        if size_test[0] > 0:
            res.append(T.Resize(size_test[0] if len(size_test) == 1 else size_test, interpolation=3))
        if do_crop:
            res.append(T.CenterCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size))
        res.append(ToTensor())
    return T.Compose(res)
