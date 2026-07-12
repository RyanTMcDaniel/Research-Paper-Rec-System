"""
DIAGNOSTIC 1: is the attention layer actually doing anything, or did it collapse
into uniform weighting (i.e. an expensive mean-pool)?

Pulls attn_weights from the trained encoder on real histories and reports how far
they are from uniform (1/L per paper). If weights are ~uniform, attention learned
nothing and your model IS mean-pool with extra steps + noise -- which would fully
explain why it loses to actual mean-pool. If weights are peaked/varied, attention
is doing something and the problem is elsewhere (pooling, projection, objective).

TORCH ONLY. No faiss.
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
N_SAMPLE    = 300


class UserHistoryEncoder(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, h, m):
        kpm = ~m.bool()
        # need_weights=True + average_attn_weights=True gives (B, L, L) averaged over heads
        a, w = self.attention(h, h, h, key_padding_mask=kpm,
                              need_weights=True, average_attn_weights=True)
        mask = m.unsqueeze(-1).float()
        pooled = (a * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(self.projection(pooled), p=2, dim=-1), w


def main():
    matrix = np.load(MATRIX_PATH).astype('float32')
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {p: i for i, p in enumerate(paper_ids)}
    matrix_t = F.normalize(torch.from_numpy(matrix), p=2, dim=1)

    model = UserHistoryEncoder(EMBED_DIM)
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    model.eval()

    h = pd.read_parquet(HISTORIES)
    test = h[h['split'] == 'test'].sample(N_SAMPLE, random_state=7)

    # measure: for each history, how far are the attention weights from uniform?
    # We look at the weight each key-paper receives, averaged over queries.
    deviations = []   # mean abs deviation from uniform, per history
    max_weights = []  # the single largest attention weight in each history
    by_len = {}

    for _, row in test.iterrows():
        ids = [str(x) for x in row['history_ids'] if str(x) in id_to_idx]
        L = len(ids)
        if L < 2:
            continue
        hv = matrix_t[[id_to_idx[x] for x in ids]].unsqueeze(0)
        with torch.no_grad():
            _, w = model(hv, torch.ones(1, L))
        # w: (1, L, L) -> average over query dim to get per-key importance (L,)
        key_importance = w[0].mean(dim=0).numpy()   # how much each paper is attended to
        uniform = 1.0 / L
        dev = np.mean(np.abs(key_importance - uniform))
        deviations.append(dev)
        max_weights.append(key_importance.max())
        by_len.setdefault(L, []).append(dev)

    deviations = np.array(deviations)
    max_weights = np.array(max_weights)

    print(f"analyzed {len(deviations)} histories\n")
    print(f"mean |weight - uniform| : {deviations.mean():.4f}")
    print(f"  (0.0 = perfectly uniform = attention collapsed to mean-pool)")
    print(f"  (higher = attention is actually selecting papers)\n")
    print(f"mean of max attention weight : {max_weights.mean():.4f}")
    print(f"  (for length-4 histories, uniform would be 0.25;")
    print(f"   >>0.25 means attention concentrates on specific papers)\n")

    print("deviation-from-uniform by history length:")
    for L in sorted(by_len):
        arr = np.array(by_len[L])
        print(f"  len {L}: mean_dev={arr.mean():.4f}  uniform_weight={1.0/L:.4f}  n={len(arr)}")

    print("\nINTERPRETATION:")
    print("  If mean_dev is near 0 across lengths -> attention is uniform ->")
    print("  it collapsed to mean-pool and adds only noise. That's your answer.")
    print("  If mean_dev is substantial -> attention IS selecting -> the failure")
    print("  is downstream (final pooling / projection / objective), not the attn.")


if __name__ == "__main__":
    main()