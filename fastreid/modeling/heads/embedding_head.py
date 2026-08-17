# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type,
            mlp_projection=False
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat
        self.mlp_projection = mlp_projection

        neck = []
        if embedding_dim > 0:
            if mlp_projection:
                # MLP projection: feat_dim -> embedding_dim -> ReLU -> embedding_dim
                neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
                neck.append(get_norm(norm_type, embedding_dim, bias_freeze=False))
                neck.append(nn.ReLU(inplace=True))
                neck.append(nn.Conv2d(embedding_dim, embedding_dim, 1, 1, bias=False))
                feat_dim = embedding_dim
            else:
                # Single linear projection
                neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
                feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim       = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim  = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes    = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat      = cfg.MODEL.HEADS.NECK_FEAT
        pool_type      = cfg.MODEL.HEADS.POOL_LAYER
        cls_type       = cfg.MODEL.HEADS.CLS_LAYER
        scale          = cfg.MODEL.HEADS.SCALE
        margin         = cfg.MODEL.HEADS.MARGIN
        with_bnneck    = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type      = cfg.MODEL.HEADS.NORM
        mlp_projection = cfg.MODEL.HEADS.MLP_PROJECTION
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type,
            'mlp_projection': mlp_projection
        }

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after': feat = neck_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": feat,
        }

@REID_HEADS_REGISTRY.register()
class PCBEmbeddingHead(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type,
            mlp_projection=False,
            num_parts=4
    ):
        """
        NOTE: this interface is experimental.
        """
        super().__init__()
        self.num_parts = num_parts
        self.neck_feat = neck_feat

        self.bottlenecks = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.weights = nn.ParameterList()

        assert hasattr(any_softmax, cls_type)

        for i in range(num_parts):
            neck = []
            cur_feat_dim = feat_dim
            if embedding_dim > 0:
                if mlp_projection:
                    neck.append(nn.Conv2d(cur_feat_dim, embedding_dim, 1, 1, bias=False))
                    neck.append(get_norm(norm_type, embedding_dim, bias_freeze=False))
                    neck.append(nn.ReLU(inplace=True))
                    neck.append(nn.Conv2d(embedding_dim, embedding_dim, 1, 1, bias=False))
                    cur_feat_dim = embedding_dim
                else:
                    neck.append(nn.Conv2d(cur_feat_dim, embedding_dim, 1, 1, bias=False))
                    cur_feat_dim = embedding_dim

            if with_bnneck:
                neck.append(get_norm(norm_type, cur_feat_dim, bias_freeze=True))

            self.bottlenecks.append(nn.Sequential(*neck))
            
            w = nn.Parameter(torch.Tensor(num_classes, cur_feat_dim))
            cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)
            self.weights.append(w)
            self.classifiers.append(cls_layer)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for b in self.bottlenecks:
            b.apply(weights_init_kaiming)
        for w in self.weights:
            nn.init.normal_(w, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        ret = EmbeddingHead.from_config(cfg)
        ret['num_parts'] = getattr(cfg.MODEL.HEADS, 'NUM_PARTS', 4)
        return ret

    def forward(self, features, targets=None):
        pool_features = F.adaptive_avg_pool2d(features, (self.num_parts, 1))

        eval_feats = []
        cls_outputs_list = []
        pred_class_logits_list = []
        train_feats = []

        for i in range(self.num_parts):
            f_i = pool_features[:, :, i:i+1, :]
            neck_f_i = self.bottlenecks[i](f_i)
            neck_f_i_flat = neck_f_i.squeeze(-1).squeeze(-1)

            if self.neck_feat == 'before':
                f = f_i.squeeze(-1).squeeze(-1)
            elif self.neck_feat == 'after':
                f = neck_f_i_flat
            else:
                f = neck_f_i_flat

            eval_f = neck_f_i_flat
            if not self.training:
                eval_f = F.normalize(eval_f, p=2, dim=1)
            eval_feats.append(eval_f)

            if self.training:
                cls_layer = self.classifiers[i]
                w = self.weights[i]
                if cls_layer.__class__.__name__ == 'Linear':
                    logits = F.linear(neck_f_i_flat, w)
                else:
                    logits = F.linear(F.normalize(neck_f_i_flat), F.normalize(w))

                cls_outputs = cls_layer(logits.clone(), targets)
                cls_outputs_list.append(cls_outputs)
                pred_class_logits_list.append(logits.mul(cls_layer.s))
                train_feats.append(f)

        if not self.training:
            return torch.cat(eval_feats, dim=1)

        return {
            "cls_outputs": cls_outputs_list,
            "pred_class_logits": pred_class_logits_list,
            "features": train_feats,
        }
