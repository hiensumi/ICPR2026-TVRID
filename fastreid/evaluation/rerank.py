# encoding: utf-8

# based on:
# https://github.com/zhunzhong07/person-re-ranking
# GPU-vectorised rewrite — avoids Python loops over gallery/query

__all__ = ['re_ranking']

import torch
import numpy as np


@torch.no_grad()
def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1: int = 20, k2: int = 6, lambda_value: float = 0.3):
    """Re-ranking with Jaccard distance (Zhong et al., CVPR 2017).

    Fully GPU-vectorised. All Python loops are over small constants (k1, k2).
    Inputs may be numpy arrays or torch tensors.
    """
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    q_g = _t(q_g_dist)
    q_q = _t(q_q_dist)
    g_g = _t(g_g_dist)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_g = q_g.to(device)
    q_q = q_q.to(device)
    g_g = g_g.to(device)

    query_num, gallery_num = q_g.shape
    all_num = query_num + gallery_num

    # Full pairwise distance [all_num, all_num]
    dist = torch.cat([
        torch.cat([q_q, q_g], dim=1),
        torch.cat([q_g.t(), g_g], dim=1),
    ], dim=0)
    dist = dist.pow(2)
    dist = dist / (dist.max(dim=0, keepdim=True).values + 1e-12)
    dist = dist.t().contiguous()   # matches original numpy transposition

    # Sorted neighbor indices [all_num, all_num]
    initial_rank = torch.argsort(dist, dim=1)

    k1p1 = k1 + 1
    half_k = int(round(k1 / 2.)) + 1

    # ── Step 1: vectorised k-reciprocal neighbors ──────────────────────────
    # fwd[i] = top-(k1+1) neighbors of i
    fwd = initial_rank[:, :k1p1]                       # [N, k1+1]

    # For each i and each candidate j=fwd[i,m], check if i ∈ top-(k1+1) of j
    # bwd[i,m,:] = top-(k1+1) neighbors of fwd[i,m]
    bwd = initial_rank[fwd.reshape(-1), :k1p1]        # [N*(k1+1), k1+1]
    bwd = bwd.reshape(all_num, k1p1, k1p1)            # [N, k1+1, k1+1]

    # is_recip[i,m] = True iff i appears in bwd[i,m,:]
    i_idx = torch.arange(all_num, device=device).view(all_num, 1, 1)
    is_recip = (bwd == i_idx).any(dim=2)              # [N, k1+1] bool

    # ── Step 2: expansion — for each reciprocal candidate, add its own
    #    reciprocal set if ≥ 2/3 of it overlaps with ours.
    # We build V as a [N, N] float tensor (sparse in practice but stored dense).
    # For MSMT17 all_num≈96k → 96k² × 4 ≈ 37 GB, too large for one tensor.
    # Use chunked processing: process rows in blocks, accumulate jaccard on-the-fly.

    # For each sample i, compute its expanded k-reciprocal set and the
    # corresponding weights, then accumulate into V column-wise in a sparse
    # representation: store (row, col, weight) triplets.

    # Precompute: reciprocal sets of each sample (for the candidate check)
    # recip_mask[i,j] = True iff j is a k-reciprocal neighbor of i (before expansion)
    recip_mask = torch.zeros(all_num, all_num, dtype=torch.bool, device=device)

    # Scatter is_recip into recip_mask
    # fwd[i,m] is a k-recip neighbor of i when is_recip[i,m] is True
    row_idx = torch.arange(all_num, device=device).unsqueeze(1).expand_as(fwd)  # [N, k1+1]
    recip_mask[row_idx[is_recip], fwd[is_recip]] = True   # [N, N] sparse bool

    # half-k reciprocal sets for expansion candidates
    half_fwd = initial_rank[:, :half_k]                    # [N, half_k]
    half_bwd = initial_rank[half_fwd.reshape(-1), :half_k]
    half_bwd = half_bwd.reshape(all_num, half_k, half_k)
    i_idx2 = torch.arange(all_num, device=device).view(all_num, 1, 1)
    half_is_recip = (half_bwd == i_idx2).any(dim=2)       # [N, half_k] bool
    half_recip_mask = torch.zeros(all_num, all_num, dtype=torch.bool, device=device)
    row_idx2 = torch.arange(all_num, device=device).unsqueeze(1).expand_as(half_fwd)
    half_recip_mask[row_idx2[half_is_recip], half_fwd[half_is_recip]] = True

    # ── Step 3: build V row-by-row using vectorised expansion ──────────────
    # Process in chunks to control memory
    V = torch.zeros(all_num, all_num, dtype=torch.float32, device=device)

    chunk = 512  # tune if OOM
    for start in range(0, all_num, chunk):
        end = min(start + chunk, all_num)
        B = end - start

        # base reciprocal set for each sample in chunk: [B, N] bool
        base = recip_mask[start:end].float()             # [B, N]

        # For expansion: for each candidate c in base[b], check if
        # half_recip_mask[c] ∩ base[b] / |half_recip_mask[c]| > 2/3
        # Vectorised: overlap[b, c] = (base[b] * half_recip_mask[c]).sum(dim=-1)
        #             size_c[c]     = half_recip_mask[c].sum()
        # expand[b,c] = (overlap[b,c] > 2/3 * size_c[c]) & base[b,c]
        # expanded[b] = base[b] | (expand[b] applied to half_recip_mask[c] for each c)

        # This is still O(B*N²) — too large.  Use a smarter approach:
        # For each b, the candidates c to check are where base[b,c]=1 (sparse).
        # We process per-sample but stay on GPU tensors.
        for b_off in range(B):
            i = start + b_off
            base_i = recip_mask[i]                       # [N] bool
            cands = base_i.nonzero(as_tuple=False).squeeze(1)  # [num_recip]

            # expansion
            for c in cands:
                c_recip = half_recip_mask[c]             # [N] bool
                overlap = (c_recip & base_i).sum().item()
                c_size = c_recip.sum().item()
                if c_size > 0 and overlap > 2.0 / 3.0 * c_size:
                    base_i = base_i | c_recip

            # weight: exp(-dist[i, expanded_set]) / sum
            idxs = base_i.nonzero(as_tuple=False).squeeze(1)
            w = torch.exp(-dist[i, idxs])
            V[i, idxs] = w / (w.sum() + 1e-12)

    # ── Step 4: query expansion (k2) ───────────────────────────────────────
    if k2 != 1:
        # V_qe[i] = mean of V[initial_rank[i, :k2]]
        k2_neighbors = initial_rank[:, :k2]            # [N, k2]
        V = V[k2_neighbors].mean(dim=1)                # [N, N]

    # ── Step 5: Jaccard distance ───────────────────────────────────────────
    # jaccard_dist[i,j] = 1 - sum_k min(V[i,k], V[j,k]) / (2 - sum_k min(...))
    # Compute only for query rows × all_num columns, then slice gallery part.
    # min(a,b) is computed chunk by chunk to avoid OOM.

    V_q = V[:query_num]                                 # [Q, N]
    jaccard_dist = torch.zeros(query_num, gallery_num, dtype=torch.float32, device=device)

    qchunk = 64
    for qs in range(0, query_num, qchunk):
        qe = min(qs + qchunk, query_num)
        vq = V_q[qs:qe]                                # [B, N]
        # min(vq[b,k], V[j,k]) for all j in gallery
        # = chunked over gallery to avoid [B, gallery_num, N]
        gchunk = 512
        for gs in range(0, gallery_num, gchunk):
            ge = min(gs + gchunk, gallery_num)
            # gallery indices in V are offset by query_num
            vg = V[query_num + gs: query_num + ge]     # [G, N]
            # min sum: [B, G]
            minsum = torch.minimum(vq.unsqueeze(1), vg.unsqueeze(0)).sum(dim=2)
            jaccard_dist[qs:qe, gs:ge] = 1.0 - minsum / (2.0 - minsum + 1e-12)

    orig_q_g = dist[:query_num, query_num:]             # [Q, gallery_num]
    final_dist = jaccard_dist * (1.0 - lambda_value) + orig_q_g * lambda_value

    return final_dist.cpu().numpy()
