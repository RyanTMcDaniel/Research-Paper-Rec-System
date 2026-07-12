"""
DIAGNOSTIC 2: did training actually accomplish anything?

Compares the TRAINED encoder against a RANDOM-INIT encoder (same architecture,
untrained weights) on Recall@10. This is the single most diagnostic test you can
run, because it cleanly separates your two hypotheses:

  * trained ~= random  -> training did essentially NOTHING. The problem is the
                          training setup (objective too easy, LR, whatever), not
                          the architecture. Fix training.
  * trained >> random  -> training worked; the model learned something real, and
                          the ceiling is ARCHITECTURAL (attention just isn't the
                          right tool for 3-5 item histories). Redesign, don't retrain.

Also compares both against mean-pool as the reference baseline.

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

def topk(q, matrix_t, paper_ids, k, exclude):
    d = torch.cdist(q.unsqueeze(0), matrix_t).squeeze(0)
    d[list(exclude)] = float('inf')
    idx = torch.topk(d, k, largest=False).indices.cpu().numpy()
    return [paper_ids[i] for i in idx]


def eval_encoder(model, test, matrix_t, id_to_idx, paper_ids):
    recalls = []
    for _, row in test.iterrows():
        ids = [str(x) for x in row['history_ids'] if str(x) in id_to_idx]
        if not ids:
            continue
        pos = str(row['positive_id'])
        idxs = [id_to_idx[x] for x in ids]
        hv = matrix_t[idxs]
        L = hv.shape[0]
        with torch.no_grad():
            fp, _ = model(hv.unsqueeze(0), torch.ones(1, L))
        r = topk(fp.squeeze(0), matrix_t, paper_ids, K, set(idxs))
        recalls.append(recall_at_k(r, pos, K))
    return float(np.mean(recalls))


def eval_meanpool(test, matrix_t, id_to_idx, paper_ids):
    recalls = []
    for _, row in test.iterrows():
        ids = [str(x) for x in row['history_ids'] if str(x) in id_to_idx]
        if not ids:
            continue
        pos = str(row['positive_id'])
        idxs = [id_to_idx[x] for x in ids]
        mp = F.normalize(matrix_t[idxs].mean(0), p=2, dim=-1)
        r = topk(mp, matrix_t, paper_ids, K, set(idxs))
        recalls.append(recall_at_k(r, pos, K))
    return float(np.mean(recalls))


def main():
    matrix = np.load(MATRIX_PATH).astype('float32')
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {p: i for i, p in enumerate(paper_ids)}
    matrix_t = F.normalize(torch.from_numpy(matrix), p=2, dim=1)

    h = pd.read_parquet(HISTORIES)
    test = h[h['split'] == 'test'].sample(min(N_EVAL, (h['split']=='test').sum()), random_state=42)
    print(f"evaluating on {len(test)} test users\n")

    # trained
    trained = UserHistoryEncoder(EMBED_DIM)
    trained.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    trained.eval()
    r_trained = eval_encoder(trained, test, matrix_t, id_to_idx, paper_ids)

    # random init (same arch, fresh weights) -- average a few seeds for stability
    rand_scores = []
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        rnd = UserHistoryEncoder(EMBED_DIM)
        rnd.eval()
        rand_scores.append(eval_encoder(rnd, test, matrix_t, id_to_idx, paper_ids))
    r_random = float(np.mean(rand_scores))

    r_mp = eval_meanpool(test, matrix_t, id_to_idx, paper_ids)

    print(f"{'encoder':<20} {'Recall@10':>10}")
    print("-" * 32)
    print(f"{'trained model':<20} {r_trained:>10.4f}")
    print(f"{'RANDOM-init model':<20} {r_random:>10.4f}   (avg of 3 seeds)")
    print(f"{'mean-pool baseline':<20} {r_mp:>10.4f}")

    print("\nVERDICT:")
    lift = r_trained - r_random
    print(f"  trained - random = {lift:+.4f}")
    if lift < 0.002:
        print("  -> training barely moved the needle vs random weights.")
        print("     The TRAINING is the problem, not the architecture. The objective")
        print("     is too easy / uninformative to teach the attention anything useful.")
    else:
        print("  -> training clearly beat random init. The model DID learn something;")
        print("     the ceiling is architectural (attention wrong for short histories).")
    print(f"\n  Either way, mean-pool ({r_mp:.4f}) is the bar. If neither trained nor")
    print(f"  random beats it, pooling is simply the right inductive bias here.")


if __name__ == "__main__":
    main()