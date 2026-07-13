"""
Experiment 4 (ABLATION.md): rule out a space mismatch between user
fingerprints and the paper index.

The projection was trained to map user-history vectors TOWARD raw paper
embeddings -- the positives and negatives in the triplet loss are raw cache
vectors -- so searching the raw paper index should be correct by
construction. This script tests the alternative anyway, because "you
evaluated in the wrong space" is the first objection anyone raises when a
trained model loses to an average: push every paper vector through the same
trained projection and search projected-against-projected.

  * projected-space search improves things -> the eval had a space bug and
    every prior number is suspect.
  * projected-space search degrades or collapses -> raw-space search was
    correct all along, and the space-mismatch explanation is eliminated.

Observed result (see ABLATION.md, Experiment 4): projected-space search
collapsed to 0.0000. Raw-space search was correct.

TORCH ONLY. No faiss. No retraining.
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
        return F.normalize(self.projection(pooled), p=2, dim=-1), _


def recall_at_k(ranked, rel, k=10):
    return 1.0 if rel in ranked[:k] else 0.0

def ndcg_at_k(ranked, rel, k=10):
    topk = list(ranked[:k])
    if rel not in topk:
        return 0.0
    return 1.0 / np.log2(topk.index(rel) + 2)

def topk(q, mat, paper_ids, k, exclude):
    d = torch.cdist(q.unsqueeze(0), mat).squeeze(0)
    d[list(exclude)] = float('inf')
    idx = torch.topk(d, k, largest=False).indices.cpu().numpy()
    return [paper_ids[i] for i in idx]


def main():
    matrix = np.load(MATRIX_PATH).astype('float32')
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {p: i for i, p in enumerate(paper_ids)}

    raw_t = F.normalize(torch.from_numpy(matrix), p=2, dim=1)   # raw space (what we used before)

    model = UserHistoryEncoder(EMBED_DIM)
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    model.eval()

    # PROJECT every paper vector through the trained projection, then normalize.
    # This is the shared space the user fingerprints actually live in.
    with torch.no_grad():
        proj_t = model.projection(raw_t)                 # (N, 256) in projected space
        proj_t = F.normalize(proj_t, p=2, dim=1)
    print("projected paper matrix:", tuple(proj_t.shape))

    h = pd.read_parquet(HISTORIES)
    test = h[h['split'] == 'test'].sample(min(N_EVAL, (h['split']=='test').sum()), random_state=42)
    print(f"evaluating {len(test)} test users\n")

    # configs:
    #  model_proj : model fingerprint  vs PROJECTED paper index   <- the fix
    #  model_raw  : model fingerprint  vs RAW paper index          <- what we did before
    #  meanpool   : raw mean           vs RAW paper index          <- baseline
    scores = {m: {'r': [], 'n': []} for m in ['model_proj', 'model_raw', 'meanpool']}

    for _, row in test.iterrows():
        ids = [str(x) for x in row['history_ids'] if str(x) in id_to_idx]
        if not ids:
            continue
        pos = str(row['positive_id'])
        idxs = [id_to_idx[x] for x in ids]
        excl = set(idxs)
        hv = raw_t[idxs].unsqueeze(0)
        L = len(idxs)

        with torch.no_grad():
            fp, _ = model(hv, torch.ones(1, L))   # projected-space fingerprint

        # fix: search the PROJECTED index
        r = topk(fp.squeeze(0), proj_t, paper_ids, K, excl)
        scores['model_proj']['r'].append(recall_at_k(r, pos, K))
        scores['model_proj']['n'].append(ndcg_at_k(r, pos, K))

        # old way: search the RAW index
        r2 = topk(fp.squeeze(0), raw_t, paper_ids, K, excl)
        scores['model_raw']['r'].append(recall_at_k(r2, pos, K))
        scores['model_raw']['n'].append(ndcg_at_k(r2, pos, K))

        # baseline
        mp = F.normalize(raw_t[idxs].mean(0), p=2, dim=-1)
        r3 = topk(mp, raw_t, paper_ids, K, excl)
        scores['meanpool']['r'].append(recall_at_k(r3, pos, K))
        scores['meanpool']['n'].append(ndcg_at_k(r3, pos, K))

    print(f"{'config':<14} {'Recall@10':>10} {'NDCG@10':>10}")
    print("-" * 36)
    for m in ['model_proj', 'model_raw', 'meanpool']:
        print(f"{m:<14} {np.mean(scores[m]['r']):>10.4f} {np.mean(scores[m]['n']):>10.4f}")

    mp_proj = np.mean(scores['model_proj']['r'])
    mp_raw  = np.mean(scores['model_raw']['r'])
    base    = np.mean(scores['meanpool']['r'])
    print("\nVERDICT:")
    print(f"  searching projected space: {mp_raw:.4f} (raw) -> {mp_proj:.4f} (projected)")
    if mp_proj > base:
        print("  -> MODEL BEATS MEAN-POOL in the shared space. The model was never")
        print("     broken -- every prior eval searched the WRONG (raw) space.")
    elif mp_proj > mp_raw + 0.002:
        print("  -> projected space helps but still <= mean-pool. Partial: right space,")
        print("     model still not decisively better than averaging.")
    else:
        print("  -> projected space doesn't help. The model genuinely isn't better than")
        print("     mean-pool. Stop torturing it; write up the honest finding.")


if __name__ == "__main__":
    main()