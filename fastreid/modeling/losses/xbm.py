import torch
import torch.nn.functional as F


class XBM:
    """Cross-Batch Memory (XBM) for metric learning.

    Maintains a FIFO queue of (feature, label) pairs from recent batches,
    enabling hard negative mining across the full identity space.

    Reference: Wang et al., "Cross-Batch Memory for Embedding Learning", CVPR 2020.
    """

    def __init__(self, size: int, feat_dim: int):
        self.size = size
        self.features = torch.zeros(size, feat_dim)
        self.labels = torch.full((size,), -1, dtype=torch.long)
        self.ptr = 0
        self.is_full = False

    def to(self, device):
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        return self

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        batch_size = features.size(0)
        feats = features.detach()

        if batch_size >= self.size:
            self.features[:] = feats[-self.size:]
            self.labels[:] = labels[-self.size:]
            self.ptr = 0
            self.is_full = True
            return

        if self.ptr + batch_size > self.size:
            overflow = self.ptr + batch_size - self.size
            self.features[self.ptr:] = feats[:self.size - self.ptr]
            self.labels[self.ptr:] = labels[:self.size - self.ptr]
            self.features[:overflow] = feats[self.size - self.ptr:]
            self.labels[:overflow] = labels[self.size - self.ptr:]
            self.ptr = overflow
            self.is_full = True
        else:
            self.features[self.ptr:self.ptr + batch_size] = feats
            self.labels[self.ptr:self.ptr + batch_size] = labels
            self.ptr += batch_size
            if self.ptr >= self.size:
                self.ptr = 0
                self.is_full = True

    def get(self):
        if self.is_full:
            return self.features, self.labels
        return self.features[:self.ptr], self.labels[:self.ptr]


def xbm_circleloss(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    xbm_embedding: torch.Tensor,
    xbm_targets: torch.Tensor,
    margin: float,
    gamma: float,
) -> torch.Tensor:
    """CircleLoss computed between current batch and XBM memory bank."""
    embedding = F.normalize(embedding, dim=1)
    xbm_embedding = F.normalize(xbm_embedding, dim=1)

    # batch-to-memory similarity: (B, M)
    dist_mat = torch.matmul(embedding, xbm_embedding.t())

    B = embedding.size(0)
    M = xbm_embedding.size(0)

    is_pos = targets.view(B, 1).eq(xbm_targets.view(1, M)).float()
    is_neg = 1.0 - is_pos

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = -gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss
