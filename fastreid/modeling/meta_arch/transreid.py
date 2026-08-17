import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.losses import *
import fastreid.layers.any_softmax as any_softmax
from .build import META_ARCH_REGISTRY


def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    try:
        x = x.view(batchsize, group, -1, dim)
    except RuntimeError:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)
    return x


@META_ARCH_REGISTRY.register()
class TransReIDBaseline(nn.Module):
    """TransReID: ViT backbone with JPM (Jigsaw Patch Module) + SIE.

    Implements the architecture from He et al., ICCV 2021.
    https://arxiv.org/abs/2102.04378

    The backbone (with LOCAL_FEATURE=True) runs all transformer blocks except
    the last one, returning intermediate tokens [B, N+1, D]. Two deep copies
    of the final block + LayerNorm form the global branch (b1) and shared local
    branch (b2). JPM divides the (optionally shuffled) patch sequence into
    divide_length groups; each group is processed independently by b2.
    Training returns 5-branch outputs (1 global + 4 local). Inference returns
    the concatenation of all 5 BNNeck features, with local features scaled by 1/4.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            b1,
            b2,
            bottlenecks,
            weights,
            cls_layers,
            pixel_mean,
            pixel_std,
            neck_feat,
            loss_kwargs,
            shuffle_groups,
            shift_num,
            divide_length,
            re_arrange,
    ):
        """NOTE: this interface is experimental."""
        super().__init__()
        self.backbone = backbone
        self.b1 = b1
        self.b2 = b2
        self.bottlenecks = nn.ModuleList(bottlenecks)
        self.weights = nn.ParameterList(weights)
        self.cls_layers = nn.ModuleList(cls_layers)
        self.neck_feat = neck_feat
        self.loss_kwargs = loss_kwargs
        self.shuffle_groups = shuffle_groups
        self.shift_num = shift_num
        self.divide_length = divide_length
        self.re_arrange = re_arrange

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        last_block = backbone.blocks[-1]
        norm_layer = backbone.norm
        b1 = nn.Sequential(copy.deepcopy(last_block), copy.deepcopy(norm_layer))
        b2 = nn.Sequential(copy.deepcopy(last_block), copy.deepcopy(norm_layer))

        num_branches = 1 + cfg.MODEL.HEADS.DIVIDE_LENGTH  # global + local groups
        bottlenecks = []
        for _ in range(num_branches):
            bn = nn.BatchNorm1d(feat_dim)
            bn.bias.requires_grad_(False)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)
            bottlenecks.append(bn)

        weights = []
        for _ in range(num_branches):
            w = nn.Parameter(torch.empty(num_classes, feat_dim))
            nn.init.normal_(w, std=0.001)
            weights.append(w)

        global_cls_type = cfg.MODEL.HEADS.CLS_LAYER
        local_cls_type = cfg.MODEL.HEADS.LOCAL_CLS_LAYER or global_cls_type
        global_cls = getattr(any_softmax, global_cls_type)(num_classes, cfg.MODEL.HEADS.SCALE, cfg.MODEL.HEADS.MARGIN)
        if local_cls_type == global_cls_type:
            cls_layers = [global_cls] * num_branches
        else:
            local_cls = getattr(any_softmax, local_cls_type)(num_classes, 1.0, 0.0)
            cls_layers = [global_cls] + [local_cls] * (num_branches - 1)

        return {
            'backbone': backbone,
            'b1': b1,
            'b2': b2,
            'bottlenecks': bottlenecks,
            'weights': weights,
            'cls_layers': cls_layers,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'neck_feat': cfg.MODEL.HEADS.NECK_FEAT,
            'loss_kwargs': {
                'loss_names': cfg.MODEL.LOSSES.NAME,
                'ce': {
                    'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                    'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                    'scale': cfg.MODEL.LOSSES.CE.SCALE,
                },
                'tri': {
                    'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                    'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                    'scale': cfg.MODEL.LOSSES.TRI.SCALE,
                },
                'circle': {
                    'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                    'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                    'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE,
                },
            },
            'shuffle_groups': cfg.MODEL.HEADS.SHUFFLE_GROUPS,
            'shift_num': cfg.MODEL.HEADS.SHIFT_NUM,
            'divide_length': cfg.MODEL.HEADS.DIVIDE_LENGTH,
            're_arrange': cfg.MODEL.HEADS.RE_ARRANGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError(f"batched_inputs must be dict or Tensor, got {type(batched_inputs)}")
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        camera_id = batched_inputs.get('camids', None) if isinstance(batched_inputs, dict) else None

        # Backbone returns [B, N+1, D] intermediate tokens (LOCAL_FEATURE=True)
        tokens = self.backbone(images, camera_id=camera_id)

        # Global branch: last block + norm → CLS token
        b1_out = self.b1(tokens)          # [B, N+1, D]
        global_feat = b1_out[:, 0]        # [B, D]

        # Local branch: (optionally shuffle) → divide into groups → b2
        cls_token = tokens[:, 0:1]        # [B, 1, D]
        patch_length = (tokens.size(1) - 1) // self.divide_length

        if self.re_arrange:
            patches = shuffle_unit(tokens, self.shift_num, self.shuffle_groups)
        else:
            patches = tokens[:, 1:]       # [B, N, D]

        local_feats = []
        for i in range(self.divide_length):
            group = patches[:, i * patch_length:(i + 1) * patch_length]
            group_out = self.b2(torch.cat([cls_token, group], dim=1))
            local_feats.append(group_out[:, 0])   # [B, D]

        all_feats = [global_feat] + local_feats   # 5 × [B, D]
        bn_feats = [self.bottlenecks[i](f) for i, f in enumerate(all_feats)]

        if not self.training:
            feats = bn_feats if self.neck_feat == 'after' else all_feats
            cls_names = {l.__class__.__name__ for l in self.cls_layers}
            if len(cls_names) > 1:
                feats = [F.normalize(f) for f in feats]
            return torch.cat([feats[0]] + [f / 4 for f in feats[1:]], dim=1)

        targets = batched_inputs['targets']
        if targets.sum() < 0:
            targets.zero_()

        cls_outputs_list, pred_logits_list = [], []
        for i in range(len(all_feats)):
            if self.cls_layers[i].__class__.__name__ == 'Linear':
                logits = F.linear(bn_feats[i], self.weights[i])
            else:
                logits = F.linear(F.normalize(bn_feats[i]), F.normalize(self.weights[i]))
            pred_logits_list.append(logits)
            cls_outputs_list.append(self.cls_layers[i](logits.clone(), targets))

        outputs = {
            'cls_outputs': cls_outputs_list,
            'pred_class_logits': pred_logits_list,
            'features': all_feats,
        }
        return self.losses(outputs, targets)

    def losses(self, outputs, gt_labels):
        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']
        num_local = len(outputs['features']) - 1  # exclude global branch

        global_cls    = 0.0
        global_tri    = 0.0
        global_circle = 0.0
        local_cls     = 0.0
        local_tri     = 0.0
        local_circle  = 0.0

        for i, (cls_out, feat, logits) in enumerate(zip(
                outputs['cls_outputs'], outputs['features'], outputs['pred_class_logits'])):
            if i == 0:
                log_accuracy(logits.detach(), gt_labels)

            if 'CrossEntropyLoss' in loss_names:
                ce_kwargs = self.loss_kwargs['ce']
                v = cross_entropy_loss(cls_out, gt_labels, ce_kwargs['eps'], ce_kwargs['alpha']) * ce_kwargs['scale']
                if i == 0:
                    global_cls = v
                else:
                    local_cls = local_cls + v

            if 'TripletLoss' in loss_names:
                tri_kwargs = self.loss_kwargs['tri']
                v = triplet_loss(feat, gt_labels,
                                 tri_kwargs['margin'], tri_kwargs['norm_feat'], tri_kwargs['hard_mining']
                                 ) * tri_kwargs['scale']
                if i == 0:
                    global_tri = v
                else:
                    local_tri = local_tri + v

            if 'CircleLoss' in loss_names:
                circle_kwargs = self.loss_kwargs['circle']
                v = pairwise_circleloss(feat.float(), gt_labels,
                                        circle_kwargs['margin'], circle_kwargs['gamma']
                                        ) * circle_kwargs['scale']
                if i == 0:
                    global_circle = v
                else:
                    local_circle = local_circle + v

            # Hybrid: circle on global branch only
            if 'CircleLossGlobal' in loss_names and i == 0:
                circle_kwargs = self.loss_kwargs['circle']
                global_circle = pairwise_circleloss(
                    feat.float(), gt_labels, circle_kwargs['margin'], circle_kwargs['gamma']
                ) * circle_kwargs['scale']

            # Hybrid: triplet on local branches only
            if 'TripletLossLocal' in loss_names and i > 0:
                tri_kwargs = self.loss_kwargs['tri']
                local_tri = local_tri + triplet_loss(
                    feat, gt_labels, tri_kwargs['margin'], tri_kwargs['norm_feat'], tri_kwargs['hard_mining']
                ) * tri_kwargs['scale']

        # 50/50 weighting: global branch = 0.5, local branch average = 0.5
        if 'CrossEntropyLoss' in loss_names:
            loss_dict['loss_cls'] = 0.5 * global_cls + 0.5 * (local_cls / num_local)
        if 'TripletLoss' in loss_names:
            loss_dict['loss_triplet'] = 0.5 * global_tri + 0.5 * (local_tri / num_local)
        if 'CircleLoss' in loss_names:
            loss_dict['loss_circle'] = 0.5 * global_circle + 0.5 * (local_circle / num_local)

        # Hybrid losses: each covers only its assigned branches, no 50/50 penalty
        if 'CircleLossGlobal' in loss_names:
            loss_dict['loss_circle_global'] = global_circle
        if 'TripletLossLocal' in loss_names:
            loss_dict['loss_triplet_local'] = local_tri / num_local

        return loss_dict
