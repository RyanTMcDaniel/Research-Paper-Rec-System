"""
DIAGNOSTIC 3: component ablation. We've proven (a) training worked (143x over
random) and (b) attention actually selects papers (non-uniform weights). Yet the
model still loses to mean-pool. So the damage is in a component neither prior test
isolated -- prime suspect: the learned PROJECTION layer, which mean-pool skips.

Evaluates FOUR configs on the SAME test users, same normalized geometry:

  1. full          : attention -> pool -> projection -> normalize   (your model)
  2. no_projection : attention -> pool -> normalize                  (drop the linear map)
  3. no_attention  : raw mean -> projection -> normalize             (keep proj, kill attn)
  4. meanpool      : raw mean -> normalize                           (the baseline that wins)

Cross-reference the four:
  * if (2) >> (1)  -> the PROJECTION is dragging you down. Kill/rework it.
  * if (3) ~ (1)   -> attention adds nothing; the projection is doing the work (badly).
  * if (2) ~ (4)   -> without the projection you basically ARE mean-pool -> proj is the whole gap.

TORCH ONLY. No faiss. No retraining -- all four reuse the trained weights.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE        = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES   = os.path.join(BASE, "histories.parquet")
MATRIX_PATH = os.path.join(BASE, "paper_matrix.npy")
IDS_PATH    = os.path.join(BASE, "paper_ids.npy")
CKPT_PATH   = os.path.join(REPO_ROOT, "scripts", "models", "user_history_encoder_best.pt")
EMBED_DIM   = 256
K           = 10
N_EVAL      = 3000


class UserHistoryEncoder(nn.Module):
    """Same arch as trained. forward() takes a `mode` so we can ablate components
    while reusing the exact trained weights -- no retraining needed."""
    def __init__(self, embedding_dim=256, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def _pool(self, seq, m):
        mask = m.unsqueeze(-1).float()
        return (seq * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

    def forward(self, h, m, mode="full"):
        if mode in ("full", "no_projection"):
            kpm = ~m.bool()
            seq, _ = self.attention(h, h, h, key_padding_mask=kpm)
            pooled = self._pool(seq, m)
        else:  # no_attention / meanpool -> pool the raw history vectors
            pooled = self._pool(h, m)

        if mode in ("full", "no_attention"):
            out = self.projection(pooled)
        else:  # no_projection / meanpool -> skip the linear map
            out = pooled

        return F.normalize(out, p=2, dim=-1)


def recall_at_k(ranked, rel, k=10):
    return 1.0 if rel in ranked[:k] else 0.0

def ndcg_at_k(ranked, rel, k=10):
    topk = list(ranked[:k])
    if rel not in topk:
        return 0.0
    return 1.0 / np.log2(topk.index(rel) + 2)

def topk(q, matrix_t, paper_ids, k, exclude):
    d = torch.cdist(q.unsqueeze(0), matrix_t).squeeze(0)
    d[list(exclude)] = float('inf')
    idx = torch.topk(d, k, largest=False).indices.cpu().numpy()
    return [paper_ids[i] for i in idx]


def main():
    matrix = np.load(MATRIX_PATH).astype('float32')
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {p: i for i, p in enumerate(paper_ids)}
    matrix_t = F.normalize(torch.from_numpy(matrix), p=2, dim=1)

    model = UserHistoryEncoder(EMBED_DIM)
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    model.eval()

    h = pd.read_parquet(HISTORIES)
    test = h[h['split'] == 'test'].sample(min(N_EVAL, (h['split']=='test').sum()), random_state=42)
    print(f"evaluating {len(test)} test users across 4 ablation configs\n")

    modes = ["full", "no_projection", "no_attention", "meanpool"]
    scores = {m: {'r': [], 'n': []} for m in modes}

    for _, row in test.iterrows():
        ids = [str(x) for x in row['history_ids'] if str(x) in id_to_idx]
        if not ids:
            continue
        pos = str(row['positive_id'])
        idxs = [id_to_idx[x] for x in ids]
        hv = matrix_t[idxs].unsqueeze(0)
        L = len(idxs)
        ones = torch.ones(1, L)
        excl = set(idxs)
        for mode in modes:
            with torch.no_grad():
                fp = model(hv, ones, mode=mode).squeeze(0)
            ranked = topk(fp, matrix_t, paper_ids, K, excl)
            scores[mode]['r'].append(recall_at_k(ranked, pos, K))
            scores[mode]['n'].append(ndcg_at_k(ranked, pos, K))

    print(f"{'config':<16} {'Recall@10':>10} {'NDCG@10':>10}")
    print("-" * 38)
    for mode in modes:
        r = np.mean(scores[mode]['r'])
        n = np.mean(scores[mode]['n'])
        print(f"{mode:<16} {r:>10.4f} {n:>10.4f}")

    full = np.mean(scores['full']['r'])
    nop  = np.mean(scores['no_projection']['r'])
    noa  = np.mean(scores['no_attention']['r'])
    mp   = np.mean(scores['meanpool']['r'])

    print("\nVERDICT:")
    print(f"  dropping projection: {full:.4f} -> {nop:.4f}  ({nop-full:+.4f})")
    print(f"  dropping attention : {full:.4f} -> {noa:.4f}  ({noa-full:+.4f})")
    if nop > full + 0.001:
        print("  -> the PROJECTION layer is hurting you. The pooled vector is better")
        print("     BEFORE the learned linear map mangles it. Rework/remove projection.")
    if abs(nop - mp) < 0.002:
        print("  -> no_projection ~= mean-pool: the projection accounts for basically")
        print("     the entire gap to the baseline.")
    if noa > full + 0.001:
        print("  -> the ATTENTION is hurting: raw-mean-into-projection beats full model.")


if __name__ == "__main__":
    main()