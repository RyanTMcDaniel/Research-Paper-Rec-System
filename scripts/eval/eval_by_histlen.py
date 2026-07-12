"""
Length-stratified eval: does attention's deficit vs mean-pool shrink as history
length grows? Hypothesis: attention adds little on 3-item histories (nothing to
reweight) but should help more on 5-item ones. If the gap narrows with length,
that's direct evidence the architecture is mismatched to SHORT sequences, not
broken in general.

TORCH ONLY. Reuses the normalized geometry from evaluate.py.
Buckets test users by history_len and reports Recall@10 / NDCG@10 per bucket
for both the trained model and the mean-pool baseline.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ---------- PATHS ----------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE        = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES   = os.path.join(BASE, "histories.parquet")   # or your new-rec-data local copy if you have one
MATRIX_PATH = os.path.join(BASE, "paper_matrix.npy")
IDS_PATH    = os.path.join(BASE, "paper_ids.npy")
CKPT_PATH   = os.path.join(REPO_ROOT, "scripts", "models", "user_history_encoder_best.pt")  # the hard-neg normalized ckpt
EMBED_DIM   = 256
K           = 10
N_EVAL_USERS = 4000   # a bit larger so each length bucket has enough samples


class UserHistoryEncoder(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, h, m):
        kpm = ~m.bool()
        a, _ = self.attention(h, h, h, key_padding_mask=kpm)
        mask = m.unsqueeze(-1).float()
        pooled = (a * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        out = self.projection(pooled)
        return F.normalize(out, p=2, dim=-1), _


def recall_at_k(ranked, rel, k=10):
    return 1.0 if rel in ranked[:k] else 0.0

def ndcg_at_k(ranked, rel, k=10):
    topk = list(ranked[:k])
    if rel not in topk:
        return 0.0
    return 1.0 / np.log2(topk.index(rel) + 2)

def topk_by_l2(q, matrix_t, paper_ids, k, exclude=None):
    d = torch.cdist(q.unsqueeze(0), matrix_t).squeeze(0)
    if exclude:
        d[list(exclude)] = float('inf')
    idx = torch.topk(d, k, largest=False).indices.cpu().numpy()
    return [paper_ids[i] for i in idx]


def main():
    device = torch.device('cpu')

    matrix = np.load(MATRIX_PATH).astype('float32')
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {p: i for i, p in enumerate(paper_ids)}
    matrix_t = F.normalize(torch.from_numpy(matrix), p=2, dim=1)   # unit sphere
    print(f"paper space: {tuple(matrix_t.shape)} (normalized)")

    model = UserHistoryEncoder(EMBED_DIM).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    h = pd.read_parquet(HISTORIES)
    test = h[h['split'] == 'test'].reset_index(drop=True)
    if N_EVAL_USERS:
        test = test.sample(n=min(N_EVAL_USERS, len(test)), random_state=42).reset_index(drop=True)
    print(f"evaluating {len(test)} test users, bucketed by history length\n")

    # buckets: {hist_len: {'model_r':[], 'model_n':[], 'mp_r':[], 'mp_n':[]}}
    buckets = defaultdict(lambda: defaultdict(list))

    for _, row in test.iterrows():
        hist_ids = [str(x) for x in row['history_ids']]
        positive = str(row['positive_id'])
        hist_idx = [id_to_idx[x] for x in hist_ids if x in id_to_idx]
        if not hist_idx:
            continue
        exclude = set(hist_idx)
        hv = matrix_t[hist_idx]
        L = hv.shape[0]   # actual usable history length

        # MODEL
        with torch.no_grad():
            fp, _ = model(hv.unsqueeze(0), torch.ones(1, L))
        r_model = topk_by_l2(fp.squeeze(0), matrix_t, paper_ids, K, exclude)
        buckets[L]['model_r'].append(recall_at_k(r_model, positive, K))
        buckets[L]['model_n'].append(ndcg_at_k(r_model, positive, K))

        # MEAN-POOL
        mp = F.normalize(hv.mean(0), p=2, dim=-1)
        r_mp = topk_by_l2(mp, matrix_t, paper_ids, K, exclude)
        buckets[L]['mp_r'].append(recall_at_k(r_mp, positive, K))
        buckets[L]['mp_n'].append(ndcg_at_k(r_mp, positive, K))

    # report
    print(f"{'histlen':>7} {'n':>6} {'model_R':>9} {'mp_R':>9} {'gap_R':>9} {'model_N':>9} {'mp_N':>9}")
    print("-" * 62)
    for L in sorted(buckets):
        b = buckets[L]
        n = len(b['model_r'])
        mr, pr = np.mean(b['model_r']), np.mean(b['mp_r'])
        mn, pn = np.mean(b['model_n']), np.mean(b['mp_n'])
        gap = mr - pr   # negative = model worse; closer to 0 = attention catching up
        print(f"{L:>7} {n:>6} {mr:>9.4f} {pr:>9.4f} {gap:>+9.4f} {mn:>9.4f} {pn:>9.4f}")

    print("\nRead gap_R: negative = model loses to mean-pool. If gap_R rises toward 0 "
          "(or flips positive) as histlen grows, attention's deficit shrinks with length.")


if __name__ == "__main__":
    main()