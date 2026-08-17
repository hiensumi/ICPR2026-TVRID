import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassMemory(nn.Module):
    """Class-Level Feature Memory with EMA updates.

    Maintains one L2-normalized prototype per identity. Each prototype is
    updated as an exponential moving average of the batch features belonging
    to that class:

        p_c = momentum * p_c + (1 - momentum) * mean(batch_features_of_class_c)
        p_c = normalize(p_c)

    Prototypes and initialization mask are registered as buffers so they
    are saved in state_dict() and restored on resume — matching the
    persistence behavior of SpCL / CAP HybridMemory implementations.

    Unlike XBM, which stores stale instance features in a FIFO queue and
    therefore breaks CircleLoss's adaptive weighting, CLM stores a small,
    smoothly-evolving set of class centers. This avoids the staleness
    problem while still exposing every identity as a candidate negative in
    every batch.
    """

    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.999):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        feats = F.normalize(features.detach(), dim=1)
        unique_labels = torch.unique(labels)
        for c in unique_labels:
            mask = labels == c
            mean_feat = feats[mask].mean(dim=0)
            mean_feat = F.normalize(mean_feat, dim=0)
            c_int = c.item()
            if not self.initialized[c_int]:
                self.prototypes[c_int] = mean_feat
                self.initialized[c_int] = True
            else:
                p = self.momentum * self.prototypes[c_int] + (1.0 - self.momentum) * mean_feat
                self.prototypes[c_int] = F.normalize(p, dim=0)

    def get(self):
        """Return only initialized prototypes and their class indices."""
        if self.initialized.any():
            idx = torch.nonzero(self.initialized, as_tuple=False).squeeze(1)
            return self.prototypes[idx], idx
        return self.prototypes[:0], self.initialized.new_zeros(0, dtype=torch.long)

    def num_initialized(self) -> int:
        return int(self.initialized.sum().item())


def clm_circleloss(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    proto_embedding: torch.Tensor,
    proto_labels: torch.Tensor,
    margin: float,
    gamma: float,
) -> torch.Tensor:
    """CircleLoss between batch features and class-level prototypes.

    Shape: embedding (B, D), proto_embedding (C_init, D), where C_init is
    the number of identities that have been observed at least once.
    """
    embedding = F.normalize(embedding, dim=1)
    # prototypes are already normalized inside ClassMemory.update

    dist_mat = torch.matmul(embedding, proto_embedding.t())  # (B, C_init)

    B = embedding.size(0)
    C = proto_embedding.size(0)

    is_pos = targets.view(B, 1).eq(proto_labels.view(1, C)).float()
    is_neg = 1.0 - is_pos

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = -gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

    # Rows where this batch sample has no matching prototype (identity never
    # seen yet) will have -inf in logit_p; mask them out.
    has_pos = is_pos.sum(dim=1) > 0
    if not has_pos.any():
        return embedding.new_zeros(())

    lse_p = torch.logsumexp(logit_p[has_pos], dim=1)
    lse_n = torch.logsumexp(logit_n[has_pos], dim=1)
    loss = F.softplus(lse_p + lse_n).mean()
    return loss
