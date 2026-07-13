"""
Cold-start evaluation harness.

Foundation + seed-paper method + category-selection method + convergence sweep,
all on ONE fixed population so the two methods are directly comparable.

DESIGN (locked):
  - Option 1: single held-out positive (same axis as main eval).
  - Option A fixed population: length-FIXED_LEN users who ALSO have
    >= MIN_CATS distinct categories in their signal pool. Same users support
    the seed-paper sweep {1..FIXED_LEN-1} AND the category sweep {1..MIN_CATS}.
  - Seed-paper: N real papers -> tower (and mean-pool). Sweep {1,2,3,4}.
  - Category:   N top category CENTROIDS fed as an N-item pseudo-history ->
    tower (and mean-pool). Sweep {1,2,3}. (Bridge 1: N centroids as N-item
    history, so processing is IDENTICAL to seed-paper; only the input source
    differs — real paper vecs vs field-level centroids.)
  - Category sweep caps at MIN_CATS because category diversity thins faster
    than paper count; seed-paper extends one level further (4 papers always
    available). They overlap on {1,2,3} where directly comparable.

GEOMETRY: model outputs unit-norm fingerprints; paper matrix + centroids unit-
norm. Every query vector is L2-normalized before topk_by_l2. Non-negotiable.

Categories load from the FLAT paper_categories.parquet (paperId -> list[str]).
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter

from evaluate import (
    UserHistoryEncoder,
    topk_by_l2,
    recall_at_k,
    ndcg_at_k,
    EMBED_DIM,
    K,
)


# =============================================================================
# CONFIG
# =============================================================================
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE            = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES       = os.path.join(BASE, "histories.parquet")
MATRIX_PATH     = os.path.join(BASE, "paper_matrix.npy")
IDS_PATH        = os.path.join(BASE, "paper_ids.npy")
CENTROIDS_PATH  = os.path.join(BASE, "category_centroids.parquet")
CATEGORIES_PATH = os.path.join(BASE, "paper_categories.parquet")   # flat paperId -> list[str]
CORPUS_PATH     = os.path.join(BASE, "cleaned_corpus.parquet")      # for popularity (citationCount)
CKPT_PATH       = os.path.join(REPO_ROOT, "scripts", "models", "user_history_encoder_best.pt")

FIXED_LEN       = 5             # fixed history length
MIN_CATS        = 3            # users must have >= this many distinct categories
PAPER_LEVELS    = (1, 2, 3, 4)  # seed-paper sweep (max = FIXED_LEN - 1)
CAT_LEVELS      = (1, 2, 3)     # category sweep (capped by diversity = MIN_CATS)
N_EVAL_USERS    = 10000         # subsample of the intersection population
SEED            = 1234
# =============================================================================


def l2norm_np(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.clip(n, 1e-9, None)).astype(np.float32)


# =============================================================================
# ARTIFACTS
# =============================================================================
def load_artifacts(device="cpu"):
    matrix = np.load(MATRIX_PATH).astype("float32")
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    matrix_t = F.normalize(torch.from_numpy(matrix).to(device), p=2, dim=1)
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    tower = UserHistoryEncoder(embedding_dim=EMBED_DIM).to(device)
    tower.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    tower.eval()

    cdf = pd.read_parquet(CENTROIDS_PATH)
    dim_cols = [c for c in cdf.columns if c.startswith("dim_")]
    assert len(dim_cols) == EMBED_DIM, f"centroids {len(dim_cols)}d != {EMBED_DIM}"
    centroids = {row["category"]: l2norm_np(row[dim_cols].to_numpy(dtype=np.float32))
                 for _, row in cdf.iterrows()}

    print(f"paper space: {tuple(matrix_t.shape)} (L2-normalized) | "
          f"{len(centroids)} centroids | tower: {CKPT_PATH.split('/')[-1]}")
    return matrix_t, paper_ids, id_to_idx, tower, centroids


# =============================================================================
# POPULATION  (Option 1 + Option A + category-diversity intersection)
# =============================================================================
def build_coldstart_users(id_to_idx, centroids):
    rng = np.random.default_rng(SEED)

    hist = pd.read_parquet(HISTORIES)
    test = hist[hist["split"] == "test"].reset_index(drop=True)
    test = test[test["history_len"] == FIXED_LEN]

    cats_df = pd.read_parquet(CATEGORIES_PATH)
    paper_cats = {str(pid): list(c) for pid, c in
                  zip(cats_df["paperId"], cats_df["categories"])}
    n_nonempty = sum(1 for v in paper_cats.values() if v)
    print(f"[join] paper_cats: {len(paper_cats):,} entries, "
          f"{n_nonempty:,} non-empty ({100*n_nonempty/max(len(paper_cats),1):.1f}%)")

    users = []
    join_hits = join_total = 0
    dropped_diversity = 0
    for i, row in test.iterrows():
        h_ids = [str(p) for p in row["history_ids"] if str(p) in id_to_idx]
        if len(h_ids) != FIXED_LEN:
            continue

        order = rng.permutation(len(h_ids))
        h_ids = [h_ids[j] for j in order]
        positive_id = h_ids[-1]
        signal_pool = h_ids[:-1]

        for p in signal_pool:
            join_total += 1
            if p in paper_cats:
                join_hits += 1

        # ranked categories by frequency across the signal pool, keeping only
        # categories that actually have a centroid (so the method can encode them)
        c = Counter(cat for p in signal_pool for cat in paper_cats.get(p, []))
        signal_cats = [cat for cat, _ in c.most_common() if cat in centroids]

        # intersection filter: need >= MIN_CATS distinct usable categories so
        # this user supports the FULL category sweep on the SAME population
        if len(signal_cats) < MIN_CATS:
            dropped_diversity += 1
            continue

        users.append({
            "user_id":     row.get("chain_id", i),
            "signal_pool": signal_pool,   # FIXED_LEN-1 papers, fixed order
            "positive_id": positive_id,
            "signal_cats": signal_cats,   # >= MIN_CATS, ranked, all have centroids
        })

    if join_total:
        print(f"[join] signal-paper -> category hits: {join_hits:,}/{join_total:,} "
              f"({100*join_hits/join_total:.1f}%)")
    print(f"[pop] dropped {dropped_diversity:,} users with < {MIN_CATS} usable categories")

    if N_EVAL_USERS is not None and len(users) > N_EVAL_USERS:
        idx = rng.choice(len(users), size=N_EVAL_USERS, replace=False)
        users = [users[j] for j in idx]

    print(f"cold-start population: {len(users)} users "
          f"(len={FIXED_LEN}, >= {MIN_CATS} cats | paper sweep {PAPER_LEVELS} | "
          f"cat sweep {CAT_LEVELS})")
    return users


# =============================================================================
# ENCODING LAYER  --  signal -> unit-norm query vector. Reusable by the demo.
# =============================================================================

def _run_tower(vec_stack, tower):
    """N vectors (N,256) -> tower as an N-item history -> unit-norm fingerprint."""
    N = vec_stack.shape[0]
    with torch.no_grad():
        fp, _ = tower(vec_stack.unsqueeze(0), torch.ones(1, N))
    return fp.squeeze(0)


def encode_seed_papers(paper_ids_list, id_to_idx, matrix_t, tower):
    idxs = [id_to_idx[p] for p in paper_ids_list if p in id_to_idx]
    if not idxs:
        return None
    return _run_tower(matrix_t[idxs], tower)


def encode_seed_papers_meanpool(paper_ids_list, id_to_idx, matrix_t):
    idxs = [id_to_idx[p] for p in paper_ids_list if p in id_to_idx]
    if not idxs:
        return None
    return F.normalize(matrix_t[idxs].mean(dim=0), p=2, dim=-1)


def encode_categories(cat_list, centroids, tower):
    """Bridge 1: N category centroids fed as an N-item pseudo-history -> tower."""
    vecs = [centroids[c] for c in cat_list if c in centroids]
    if not vecs:
        return None
    stack = torch.tensor(np.stack(vecs), dtype=torch.float32)  # (N,256), unit-norm
    return _run_tower(stack, tower)


def encode_categories_meanpool(cat_list, centroids):
    vecs = [centroids[c] for c in cat_list if c in centroids]
    if not vecs:
        return None
    mp = np.stack(vecs).mean(axis=0)
    return F.normalize(torch.tensor(mp, dtype=torch.float32), p=2, dim=-1)


# =============================================================================
# SWEEP  --  BATCHED. For each (method, level) we build ALL users' query vectors
# into one (U, 256) matrix, do a SINGLE cdist against the 253K paper matrix, and
# extract topk per row. This replaces ~140K sequential cdist calls with a few
# dozen batched ones (10-50x faster). Metrics unchanged; only the plumbing.
#
# Metrics collected: Recall@10, NDCG@10, and Recall@100 (the wide-net metric
# that shows category is directionally-right-but-coarse rather than broken).
# =============================================================================

RECALL_WIDE = 100   # wide-net recall to characterize the coarse category signal


def _batched_eval(query_vecs, positives, excludes, matrix_t, paper_ids):
    """
    query_vecs: (U, 256) tensor of unit-norm queries
    positives:  list[str] length U — the held-out target per query
    excludes:   list[set[int]] length U — paper-index sets to mask per query
                (empty sets for the category method)
    Returns (recall@K list, ndcg@K list, recall@WIDE list), one entry per query.
    """
    U = query_vecs.shape[0]
    # single big distance matrix: (U, N)
    dists = torch.cdist(query_vecs, matrix_t)           # (U, N)

    # apply per-row excludes
    for i, ex in enumerate(excludes):
        if ex:
            dists[i, list(ex)] = float("inf")

    # top-WIDE indices per row (covers both K and WIDE)
    topw = torch.topk(dists, RECALL_WIDE, largest=False, dim=1).indices  # (U, WIDE)

    rec_k, ndcg_k, rec_wide = [], [], []
    for i in range(U):
        row = [paper_ids[j] for j in topw[i].tolist()]
        pos = positives[i]
        topk_ids = row[:K]
        rec_k.append(1.0 if pos in topk_ids else 0.0)
        if pos in topk_ids:
            ndcg_k.append(1.0 / np.log2(topk_ids.index(pos) + 2))
        else:
            ndcg_k.append(0.0)
        rec_wide.append(1.0 if pos in row else 0.0)
    return rec_k, ndcg_k, rec_wide


def run_sweep(users, id_to_idx, matrix_t, paper_ids, tower, centroids):
    methods = ("seed_tower", "seed_meanpool", "cat_tower", "cat_meanpool")
    out = {m: {"recall": {}, "ndcg": {}, "recall_wide": {}} for m in methods}

    def encode_for(name, item, L):
        """Return the query vector for one user under a given method+level, or None."""
        if name == "seed_tower":
            return encode_seed_papers(item["signal_pool"][:L], id_to_idx, matrix_t, tower)
        if name == "seed_meanpool":
            return encode_seed_papers_meanpool(item["signal_pool"][:L], id_to_idx, matrix_t)
        if name == "cat_tower":
            return encode_categories(item["signal_cats"][:L], centroids, tower)
        if name == "cat_meanpool":
            return encode_categories_meanpool(item["signal_cats"][:L], centroids)

    for name in methods:
        levels = PAPER_LEVELS if name.startswith("seed") else CAT_LEVELS
        is_seed = name.startswith("seed")
        for L in levels:
            qvecs, positives, excludes = [], [], []
            for u in users:
                v = encode_for(name, u, L)
                if v is None:
                    continue
                qvecs.append(v)
                positives.append(u["positive_id"])
                # seed-paper excludes its own seed papers; category excludes nothing
                if is_seed:
                    seeds = u["signal_pool"][:L]
                    excludes.append({id_to_idx[p] for p in seeds if p in id_to_idx})
                else:
                    excludes.append(set())
            if not qvecs:
                continue
            Q = torch.stack(qvecs)                       # (U, 256)
            rk, nk, rw = _batched_eval(Q, positives, excludes, matrix_t, paper_ids)
            out[name]["recall"][L] = rk
            out[name]["ndcg"][L] = nk
            out[name]["recall_wide"][L] = rw
    return out


def _print_table(out, metric, label=None, pop=None):
    label = label or metric
    print(f"\ncold-start convergence ({label})   [same population, both methods]")
    print(f"{'inputs':>7} | {'seed_tower':>11} {'seed_mpool':>11} | "
          f"{'cat_tower':>10} {'cat_mpool':>10}")
    print("-" * 62)
    all_levels = sorted(set(PAPER_LEVELS) | set(CAT_LEVELS))
    for L in all_levels:
        def cell(m):
            vals = out[m][metric].get(L)
            return f"{np.mean(vals):.4f}" if vals else "  -  "
        print(f"{L:>7} | {cell('seed_tower'):>11} {cell('seed_meanpool'):>11} | "
              f"{cell('cat_tower'):>10} {cell('cat_meanpool'):>10}")
    if pop is not None:
        print("-" * 62)
        print(f"{'popularity floor (signal-independent)':<40} {pop[metric]:.4f}")


def compute_popularity_floor(users, paper_ids):
    """
    POPULARITY BASELINE — the floor every cold-start method must beat. It ignores
    the cold-start signal entirely and recommends the globally most-cited papers
    to everyone. Flat across input levels (no signal used), so one number per
    metric. This answers 'does cold-start even beat recommending popular papers?'
    """
    corpus = pd.read_parquet(CORPUS_PATH)
    cite_col = "citationCount" if "citationCount" in corpus.columns else "citation_count"
    if cite_col not in corpus.columns:
        print("[pop-floor] no citation column found — skipping popularity baseline")
        return None
    pop_ranked = (corpus.sort_values(cite_col, ascending=False)["paperId"]
                  .astype(str).tolist())
    top_k    = pop_ranked[:K]
    top_wide = pop_ranked[:RECALL_WIDE]

    rk = [1.0 if u["positive_id"] in top_k else 0.0 for u in users]
    nk = []
    for u in users:
        if u["positive_id"] in top_k:
            nk.append(1.0 / np.log2(top_k.index(u["positive_id"]) + 2))
        else:
            nk.append(0.0)
    rw = [1.0 if u["positive_id"] in top_wide else 0.0 for u in users]
    return {"recall": np.mean(rk), "ndcg": np.mean(nk), "recall_wide": np.mean(rw)}


def main():
    matrix_t, paper_ids, id_to_idx, tower, centroids = load_artifacts()

    pm = matrix_t[0].norm().item()
    cn = float(np.linalg.norm(next(iter(centroids.values()))))
    print(f"geometry: paper[0]={pm:.4f} centroid={cn:.4f} "
          f"{'OK' if abs(pm-1)<1e-3 and abs(cn-1)<1e-3 else '!!! NOT UNIT NORM'}")

    users = build_coldstart_users(id_to_idx, centroids)
    if not users:
        print("no users — check FIXED_LEN / MIN_CATS")
        return

    out = run_sweep(users, id_to_idx, matrix_t, paper_ids, tower, centroids)
    pop = compute_popularity_floor(users, paper_ids)

    _print_table(out, "recall", label=f"Recall@{K}", pop=pop)
    _print_table(out, "ndcg", label=f"NDCG@{K}", pop=pop)
    _print_table(out, "recall_wide", label=f"Recall@{RECALL_WIDE}", pop=pop)


if __name__ == "__main__":
    main()