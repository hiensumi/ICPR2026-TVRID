# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from fastreid.modeling.losses.xbm import XBM, xbm_circleloss
from fastreid.modeling.losses.class_memory import ClassMemory, clm_circleloss
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None,
            freeze_backbone=False,
            xbm_cfg=None,
            clm_cfg=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
            freeze_backbone: If True, backbone stays in eval mode
            xbm_cfg: XBM config dict or None
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads

        self.loss_kwargs = loss_kwargs
        self._freeze_backbone = freeze_backbone

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.xbm = None
        if xbm_cfg and xbm_cfg['enabled']:
            self._xbm_cfg = xbm_cfg
            self.xbm_scale = xbm_cfg['scale']

        self.clm = None
        if clm_cfg and clm_cfg['enabled']:
            self._clm_cfg = clm_cfg
            self.clm_scale = clm_cfg['scale']

    def train(self, mode=True):
        """Override train to keep frozen backbone in eval mode."""
        super().train(mode)
        if self._freeze_backbone and mode:
            self.backbone.eval()
        return self

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        
        # Freeze backbone if configured
        freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()  # Set to eval mode for frozen BN
        
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'freeze_backbone': freeze_backbone,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                },
            'xbm_cfg': {
                'enabled': cfg.MODEL.LOSSES.XBM.ENABLED,
                'size': cfg.MODEL.LOSSES.XBM.SIZE,
                'start_epoch': cfg.MODEL.LOSSES.XBM.START_EPOCH,
                'scale': cfg.MODEL.LOSSES.XBM.SCALE,
            },
            'clm_cfg': {
                'enabled': cfg.MODEL.LOSSES.CLM.ENABLED,
                'momentum': cfg.MODEL.LOSSES.CLM.MOMENTUM,
                'scale': cfg.MODEL.LOSSES.CLM.SCALE,
                'min_classes': cfg.MODEL.LOSSES.CLM.MIN_CLASSES,
            }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        camera_id = batched_inputs.get("camids", None) if isinstance(batched_inputs, dict) else None

        # When backbone is frozen, run without tracking gradients for efficiency
        if self._freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images, camera_id=camera_id)
            # Clone to allow gradients through heads
            features = features.clone().requires_grad_(True) if self.training else features
        else:
            features = self.backbone(images, camera_id=camera_id)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)

            if hasattr(self, '_xbm_cfg'):
                pred_features = outputs['features']
                # Lazy init XBM on first forward (to get correct feat_dim)
                if self.xbm is None:
                    self.xbm = XBM(
                        size=self._xbm_cfg['size'],
                        feat_dim=pred_features.size(1)
                    ).to(pred_features.device)
                xbm_feats, xbm_labels = self.xbm.get()
                if xbm_feats.size(0) >= 64:
                    circle_kwargs = self.loss_kwargs.get('circle')
                    losses['loss_xbm'] = xbm_circleloss(
                        pred_features, targets,
                        xbm_feats, xbm_labels,
                        circle_kwargs.get('margin'),
                        circle_kwargs.get('gamma'),
                    ) * self.xbm_scale
                self.xbm.enqueue(pred_features, targets)

            if hasattr(self, '_clm_cfg'):
                pred_features = outputs['features']
                # Lazy init: use actual feature dim from the neck output.
                # Assigning an nn.Module to self.clm here triggers PyTorch's
                # __setattr__ which registers it as a submodule — so it will
                # appear in state_dict() from the very next checkpoint save.
                if self.clm is None:
                    self.clm = ClassMemory(
                        num_classes=self.heads.weight.size(0),
                        feat_dim=pred_features.size(1),
                        momentum=self._clm_cfg['momentum'],
                    ).to(pred_features.device)
                # Update BEFORE computing loss so current classes are present
                # as positives (prototypes are EMA-smoothed, so updating with
                # the current batch does not collapse the loss).
                self.clm.update(pred_features, targets)
                if self.clm.num_initialized() >= self._clm_cfg['min_classes']:
                    proto_feats, proto_labels = self.clm.get()
                    circle_kwargs = self.loss_kwargs.get('circle')
                    losses['loss_clm'] = clm_circleloss(
                        pred_features, targets,
                        proto_feats, proto_labels,
                        circle_kwargs.get('margin'),
                        circle_kwargs.get('gamma'),
                    ) * self.clm_scale

            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        pred_class_logits = outputs['pred_class_logits']
        if isinstance(pred_class_logits, list):
            # PCB multi-part loss
            loss_dict = {}
            loss_names = self.loss_kwargs['loss_names']
            num_parts = len(outputs['cls_outputs'])
            
            for k in ['loss_cls', 'loss_triplet', 'loss_circle', 'loss_cosface']:
                loss_dict[k] = 0.0
                
            for i in range(num_parts):
                cls_out = outputs['cls_outputs'][i]
                feat = outputs['features'][i]
                logits = outputs['pred_class_logits'][i].detach()
                
                if i == 0:
                    log_accuracy(logits, gt_labels)
                    
                if 'CrossEntropyLoss' in loss_names:
                    ce_kwargs = self.loss_kwargs.get('ce')
                    loss_dict['loss_cls'] += cross_entropy_loss(cls_out, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale')
                    
                if 'TripletLoss' in loss_names:
                    tri_kwargs = self.loss_kwargs.get('tri')
                    loss_dict['loss_triplet'] += triplet_loss(feat, gt_labels, tri_kwargs.get('margin'), tri_kwargs.get('norm_feat'), tri_kwargs.get('hard_mining')) * tri_kwargs.get('scale')
                    
                if 'CircleLoss' in loss_names:
                    circle_kwargs = self.loss_kwargs.get('circle')
                    loss_dict['loss_circle'] += pairwise_circleloss(feat, gt_labels, circle_kwargs.get('margin'), circle_kwargs.get('gamma')) * circle_kwargs.get('scale')
                    
                if 'Cosface' in loss_names:
                    cosface_kwargs = self.loss_kwargs.get('cosface')
                    loss_dict['loss_cosface'] += pairwise_cosface(feat, gt_labels, cosface_kwargs.get('margin'), cosface_kwargs.get('gamma')) * cosface_kwargs.get('scale')
            
            for k in list(loss_dict.keys()):
                if loss_dict[k] == 0.0:
                    del loss_dict[k]
                else:
                    loss_dict[k] /= num_parts
            return loss_dict

        # Original single-part block
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict
