"""
Evaluation harness: Recall@10 and NDCG@10 for the trained two-tower model
against three baselines (popularity, random, mean-pool-no-attention).

TORCH ONLY -- faiss deliberately NOT imported (avoids the torch+faiss OpenMP
crash; at 253K x 256 a torch matrix op is plenty fast for eval).

v2 CHANGE -- NORMALIZATION.
The trained model now outputs L2-normalized (unit-norm) fingerprints. So the
paper matrix and the mean-pool baseline vector MUST also be L2-normalized, or
we'd be comparing a unit vector against raw norm-~4.2 vectors and measuring
garbage (this is exactly what tanked the previous eval). With everything on the
unit sphere, L2 distance is monotonic with cosine distance -- the same geometry
the model was trained on. Model / mean-pool / retrieval space now all agree.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- PATHS ----------
REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE          = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES     = os.path.join(BASE, "histories.parquet")
CORPUS        = os.path.join(BASE, "cleaned_corpus.parquet")
MATRIX_PATH   = os.path.join(BASE, "paper_matrix.npy")
IDS_PATH      = os.path.join(BASE, "paper_ids.npy")
CKPT_PATH     = os.path.join(REPO_ROOT, "scripts", "models", "user_history_encoder_best.pt")
EMBED_DIM     = 256
K             = 10
N_EVAL_USERS  = 2000   # subsample test users for speed; None for all


# ---------- MODEL (must match training definition EXACTLY, incl. normalize) ----------
class UserHistoryEncoder(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, history_embeddings, history_mask):
        key_padding_mask = ~history_mask.bool()
        attn_output, attn_weights = self.attention(
            query=history_embeddings, key=history_embeddings,
            value=history_embeddings, key_padding_mask=key_padding_mask)
        mask = history_mask.unsqueeze(-1).float()
        summed = (attn_output * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        user_embedding = self.projection(mean_pooled)
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)  # matches training
        return user_embedding, attn_weights


# ---------- METRICS ----------
def recall_at_k(ranked_ids, relevant_id, k=10):
    """Single held-out item -> hit-rate@k: 1.0 if the true paper is in top-k."""
    return 1.0 if relevant_id in ranked_ids[:k] else 0.0


def ndcg_at_k(ranked_ids, relevant_id, k=10):
    """One relevant item: ideal DCG = 1, so NDCG = 1/log2(rank+2) if present."""
    topk = list(ranked_ids[:k])
    if relevant_id not in topk:
        return 0.0
    rank = topk.index(relevant_id)
    return 1.0 / np.log2(rank + 2)


# ---------- SEARCH (torch, no faiss) ----------
def topk_by_l2(query_vec, matrix_t, paper_ids, k, exclude_set_idx=None):
    """
    query_vec: (256,) -- assumed L2-normalized by the caller
    matrix_t:  (N, 256) -- assumed L2-normalized paper vectors
    Returns top-k paper_ids by ascending L2 distance (== descending cosine sim
    on unit vectors), excluding the user's own history papers.
    """
    dists = torch.cdist(query_vec.unsqueeze(0), matrix_t).squeeze(0)  # (N,)
    if exclude_set_idx:
        dists[list(exclude_set_idx)] = float('inf')
    topk_idx = torch.topk(dists, k, largest=False).indices.cpu().numpy()
    return [paper_ids[i] for i in topk_idx]


# ---------- SPACE-SANITY CHECK ----------
def space_check(model, matrix_t, id_to_idx, paper_ids, histories_df):
    """
    Confirm the model fingerprint and the paper vectors live in the same
    magnitude regime BEFORE trusting any Recall number. This is the check that
    caught the norm-18000 explosion last time. Everything here should now sit
    at norm ~1.0 and nearest-paper distances should be small (< ~1.5), not
    astronomically huge.
    """
    print("\n--- space sanity check ---")
    paper_norms = torch.linalg.norm(matrix_t, dim=1)
    print(f"paper vec norm: mean={paper_norms.mean():.3f} std={paper_norms.std():.3f}  (want ~1.0)")

    sample = histories_df[histories_df['split'] == 'test'].sample(5, random_state=1)
    for _, row in sample.iterrows():
        hist = [str(x) for x in row['history_ids'] if str(x) in id_to_idx]
        if not hist:
            continue
        hv = matrix_t[[id_to_idx[x] for x in hist]]
        L = hv.shape[0]
        with torch.no_grad():
            fp, _ = model(hv.unsqueeze(0), torch.ones(1, L))
        fp = fp.squeeze(0)
        mp = F.normalize(hv.mean(0), p=2, dim=-1)
        d_fp = torch.cdist(fp.unsqueeze(0), matrix_t).min().item()
        d_mp = torch.cdist(mp.unsqueeze(0), matrix_t).min().item()
        print(f"  len {L}: model_norm={fp.norm():.3f} mp_norm={mp.norm():.3f} "
              f"| nearest-paper model={d_fp:.3f} meanpool={d_mp:.3f}")
    print("--- end sanity check ---\n")


# ---------- MAIN ----------
def main():
    device = torch.device('cpu')

    # --- load + NORMALIZE paper space ---
    matrix = np.load(MATRIX_PATH).astype('float32')
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    matrix_t = torch.from_numpy(matrix).to(device)
    matrix_t = F.normalize(matrix_t, p=2, dim=1)   # <-- unit sphere, matches the model
    print(f"paper space: {tuple(matrix_t.shape)} (L2-normalized)")

    # --- load trained user encoder ---
    model = UserHistoryEncoder(embedding_dim=EMBED_DIM).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # --- test users ---
    histories_df = pd.read_parquet(HISTORIES)

    # run the sanity check FIRST -- bail mentally if norms aren't ~1.0
    space_check(model, matrix_t, id_to_idx, paper_ids, histories_df)

    test_df = histories_df[histories_df['split'] == 'test'].reset_index(drop=True)
    if N_EVAL_USERS:
        test_df = test_df.sample(n=min(N_EVAL_USERS, len(test_df)), random_state=42).reset_index(drop=True)
    print(f"evaluating on {len(test_df)} test users")

    # --- popularity baseline ---
    corpus = pd.read_parquet(CORPUS)
    cite_col = 'citationCount' if 'citationCount' in corpus.columns else 'citation_count'
    pop_ranked = corpus.sort_values(cite_col, ascending=False)['paperId'].astype(str).tolist()
    pop_top = pop_ranked[:200]

    rng = np.random.default_rng(42)
    scores = {name: {'recall': [], 'ndcg': []} for name in ['model', 'popularity', 'random', 'meanpool']}

    for _, row in test_df.iterrows():
        hist_ids = [str(h) for h in row['history_ids']]
        positive = str(row['positive_id'])
        hist_idx = [id_to_idx[h] for h in hist_ids if h in id_to_idx]
        exclude = set(hist_idx)
        if not hist_idx:
            continue
        hist_vecs = matrix_t[hist_idx]   # already normalized
        L = hist_vecs.shape[0]

        # MODEL (outputs unit-norm fingerprint)
        with torch.no_grad():
            fp, _ = model(hist_vecs.unsqueeze(0), torch.ones(1, L))
        ranked = topk_by_l2(fp.squeeze(0), matrix_t, paper_ids, K, exclude_set_idx=exclude)
        scores['model']['recall'].append(recall_at_k(ranked, positive, K))
        scores['model']['ndcg'].append(ndcg_at_k(ranked, positive, K))

        # MEAN-POOL (normalize for a fair fight on the unit sphere)
        mp = F.normalize(hist_vecs.mean(dim=0), p=2, dim=-1)
        ranked_mp = topk_by_l2(mp, matrix_t, paper_ids, K, exclude_set_idx=exclude)
        scores['meanpool']['recall'].append(recall_at_k(ranked_mp, positive, K))
        scores['meanpool']['ndcg'].append(ndcg_at_k(ranked_mp, positive, K))

        # POPULARITY
        pop_recs = [p for p in pop_top if p not in set(hist_ids)][:K]
        scores['popularity']['recall'].append(recall_at_k(pop_recs, positive, K))
        scores['popularity']['ndcg'].append(ndcg_at_k(pop_recs, positive, K))

        # RANDOM
        rand_idx = rng.choice(len(paper_ids), size=K, replace=False)
        rand_recs = [paper_ids[i] for i in rand_idx]
        scores['random']['recall'].append(recall_at_k(rand_recs, positive, K))
        scores['random']['ndcg'].append(ndcg_at_k(rand_recs, positive, K))

    print(f"\n{'method':<12} {'Recall@'+str(K):>10} {'NDCG@'+str(K):>10}")
    print("-" * 34)
    for name in ['model', 'meanpool', 'popularity', 'random']:
        r = np.mean(scores[name]['recall'])
        n = np.mean(scores[name]['ndcg'])
        print(f"{name:<12} {r:>10.4f} {n:>10.4f}")


if __name__ == "__main__":
    main()