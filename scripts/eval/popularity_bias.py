"""
POPULARITY-BIAS AUDIT (Step 17).

Question: are the recommenders just surfacing the most-cited papers to everyone,
regardless of the user's actual history? A model that only recommends popular
papers is a glorified trending list, not a personalized recommender.

Method:
  1. Bin the whole corpus into citation-count quartiles (Q1 = least cited ...
     Q4 = most cited). This is the "natural" distribution to compare against.
  2. For a sample of test users, generate top-10 recommendations from each of:
       - the trained model
       - the mean-pool baseline
       - popularity (most-cited overall) as a sanity reference
  3. Tabulate which citation quartile the RECOMMENDED papers fall into.
  4. Compare each method's recommendation distribution to the corpus baseline.
     Heavy skew toward Q4 => the method is popularity-biased.

TORCH ONLY. Local, CPU. No faiss, no retraining.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE        = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES   = os.path.join(BASE, "histories.parquet")
CORPUS      = os.path.join(BASE, "cleaned_corpus.parquet")
MATRIX_PATH = os.path.join(BASE, "paper_matrix.npy")
IDS_PATH    = os.path.join(BASE, "paper_ids.npy")
CKPT_PATH   = os.path.join(REPO_ROOT, "scripts", "models", "user_history_encoder_best.pt")
EMBED_DIM   = 256
K           = 10
N_EVAL      = 2000


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


def topk_idx(q, mat, k, exclude):
    d = torch.cdist(q.unsqueeze(0), mat).squeeze(0)
    d[list(exclude)] = float('inf')
    return torch.topk(d, k, largest=False).indices.cpu().numpy()


def main():
    matrix = np.load(MATRIX_PATH).astype('float32')
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {p: i for i, p in enumerate(paper_ids)}
    mat_t = F.normalize(torch.from_numpy(matrix), p=2, dim=1)

    model = UserHistoryEncoder(EMBED_DIM)
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    model.eval()

    # --- citation quartiles over the corpus (aligned to paper_ids order) ---
    corpus = pd.read_parquet(CORPUS)
    cite_map = dict(zip(corpus['paperId'].astype(str), corpus['citationCount'].fillna(0)))
    cites = np.array([cite_map.get(pid, 0) for pid in paper_ids], dtype=float)

    # quartile edges from the corpus distribution
    q = np.quantile(cites, [0.25, 0.50, 0.75])
    def bin_of(c):
        if c <= q[0]: return 0
        if c <= q[1]: return 1
        if c <= q[2]: return 2
        return 3
    paper_bin = np.array([bin_of(c) for c in cites])

    # corpus baseline distribution (what "unbiased" would roughly look like)
    corpus_dist = Counter(paper_bin)
    total = len(paper_bin)
    print("citation quartile edges (Q1|Q2|Q3):", q.tolist())
    print("corpus baseline distribution across quartiles:")
    for b in range(4):
        print(f"  Q{b+1}: {corpus_dist[b]/total:6.1%}")
    print()

    # --- generate recs, tally recommended-paper quartiles ---
    h = pd.read_parquet(HISTORIES)
    test = h[h['split'] == 'test'].sample(min(N_EVAL, (h['split']=='test').sum()), random_state=42)

    pop_order = np.argsort(-cites)  # most-cited first, for the popularity reference

    tally = {'model': Counter(), 'meanpool': Counter(), 'popularity': Counter()}
    n_users = 0

    for _, row in test.iterrows():
        ids = [str(x) for x in row['history_ids'] if str(x) in id_to_idx]
        if not ids:
            continue
        idxs = [id_to_idx[x] for x in ids]
        excl = set(idxs)
        hv = mat_t[idxs].unsqueeze(0)
        L = len(idxs)
        n_users += 1

        with torch.no_grad():
            fp, _ = model(hv, torch.ones(1, L))
        for i in topk_idx(fp.squeeze(0), mat_t, K, excl):
            tally['model'][paper_bin[i]] += 1

        mp = F.normalize(mat_t[idxs].mean(0), p=2, dim=-1)
        for i in topk_idx(mp, mat_t, K, excl):
            tally['meanpool'][paper_bin[i]] += 1

        pop_recs = [i for i in pop_order if i not in excl][:K]
        for i in pop_recs:
            tally['popularity'][paper_bin[i]] += 1

    print(f"recommendation quartile distribution over {n_users} users "
          f"({n_users*K} recs each):\n")
    print(f"{'method':<12} {'Q1(low)':>9} {'Q2':>9} {'Q3':>9} {'Q4(high)':>9}")
    print("-" * 52)
    print(f"{'CORPUS':<12} " + " ".join(f"{corpus_dist[b]/total:>8.1%}" for b in range(4)))
    for m in ['model', 'meanpool', 'popularity']:
        tot = sum(tally[m].values())
        print(f"{m:<12} " + " ".join(f"{tally[m][b]/tot:>8.1%}" for b in range(4)))

    print("\nINTERPRETATION:")
    print("  Compare each method's Q4 share to the corpus Q4 share (~25%).")
    print("  Much higher Q4 => that method over-recommends highly-cited papers")
    print("  (popularity bias). If model and mean-pool both skew high, the bias is")
    print("  inherited from the embedding space, not the architecture.")


if __name__ == "__main__":
    main()