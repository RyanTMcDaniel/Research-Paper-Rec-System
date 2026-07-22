"""
Contaminated-history eval: when a clean, single-topic history gets polluted with
off-topic papers, does attention degrade more gracefully than mean-pool?

This is the causal counterpart to diag_dispersion_eval.py. That script OBSERVES
naturally-dispersed histories; this one MANUFACTURES dispersion, holding everything
else fixed. Mean-pool has no way to ignore a junk paper -- it drags the centroid.
Attention, in principle, can down-weight it. If the model can't beat mean-pool even
here -- on histories we deliberately sabotaged -- then the attention layer is not
learning to gate, and the deficit is architectural rather than a data artifact.

Design:
  * Start from test users whose history is topically HOMOGENEOUS (every history
    paper shares the same primary field, per paper_categories.parquet).
  * Replace k of the history papers (k = 0, 1, 2) with papers sampled from a
    DISTANT field -- one of the 3 fields whose centroid is farthest from the user's
    home field in category_centroids.parquet.
  * The held-out target is never touched, and k=2 is a superset of k=1's swap, so
    the three conditions are a paired dose-response on the same users.
  * REPLACEMENT, not insertion: history length is constant across k, so length can't
    explain any movement in the metrics.
  * The retrieval candidate pool is held fixed across k -- we exclude the original
    history AND every contaminant at every k. Otherwise the excluded set would shift
    with k and the metrics would move for reasons that have nothing to do with the
    query vector.

Caveat worth knowing: this embedding space is anisotropic. Every one of the 23
category centroids sits within cosine distance ~0.04 of every other. "Distant field"
is a real ordering, but the absolute separation is small -- so a flat result here may
say as much about the embeddings as about the attention layer. The per-field distances
are printed below so you can judge the dose you actually administered.

TORCH ONLY -- no faiss. Reuses evaluate.py's model class, normalized geometry,
raw-space paper index, metrics, and checkpoint by direct import.
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the EXACT eval setup -- same model def, same metrics, same search, same paths.
from evaluate import (  # noqa: E402
    BASE,
    CKPT_PATH,
    EMBED_DIM,
    HISTORIES,
    IDS_PATH,
    K,
    MATRIX_PATH,
    UserHistoryEncoder,
    ndcg_at_k,
    recall_at_k,
    topk_by_l2,
)

CATEGORIES_PATH = os.path.join(BASE, "paper_categories.parquet")
CENTROIDS_PATH  = os.path.join(BASE, "category_centroids.parquet")

N_HOMOG_USERS    = 3000   # x3 conditions x2 methods. Recall@10 is ~1-2% at k=0 and falls to
                          # ~0 by k=2, so n has to be big or every gap is just hit-count jitter.
K_LEVELS         = [0, 1, 2]
N_DISTANT_FIELDS = 3      # sample contaminants from the 3 farthest eligible fields
MIN_FIELD_PAPERS = 200    # a field needs this many papers to be a usable contaminant pool
N_BOOT           = 2000   # paired bootstrap resamples for the gap CI
SEED             = 42


def boot_gap_ci(model_hits, mp_hits, n_boot=N_BOOT, seed=SEED):
    """
    95% CI on the PAIRED gap (model - meanpool), resampling users with replacement.
    If the interval straddles 0 the gap is not distinguishable from noise.
    """
    a = np.asarray(model_hits, dtype=float)
    b = np.asarray(mp_hits, dtype=float)
    n = len(a)
    if n < 2:
        return (float('nan'), float('nan'))
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def primary_field(cats):
    """A paper's primary field = its most frequent s2 category (ties -> first seen)."""
    c = Counter(list(cats))
    return c.most_common(1)[0][0] if c else None


def build_field_maps():
    """paperId -> primary field, and field -> centroid-distance-ranked list of other fields."""
    pc = pd.read_parquet(CATEGORIES_PATH)
    pc['primary'] = pc['categories'].map(primary_field)
    paper_field = dict(zip(pc['paperId'].astype(str), pc['primary']))

    field_papers = (pc.dropna(subset=['primary'])
                      .groupby('primary')['paperId']
                      .apply(lambda s: [str(x) for x in s])
                      .to_dict())

    cen = pd.read_parquet(CENTROIDS_PATH)
    fields = cen['category'].tolist()
    C = cen[[f'dim_{i}' for i in range(EMBED_DIM)]].values.astype('float32')
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    dist = 1.0 - (C @ C.T)   # cosine distance between field centroids

    # for each home field, the farthest fields that have a big enough paper pool
    distant = {}
    for i, home in enumerate(fields):
        ranked = np.argsort(-dist[i])
        picks = []
        for j in ranked:
            f = fields[j]
            if f == home:
                continue
            if len(field_papers.get(f, [])) < MIN_FIELD_PAPERS:
                continue
            picks.append((f, float(dist[i, j])))
            if len(picks) == N_DISTANT_FIELDS:
                break
        distant[home] = picks

    return paper_field, field_papers, distant


def evaluate_history(model, hist_idx, positive, exclude, matrix_t, paper_ids):
    """Model + mean-pool metrics for one (possibly contaminated) history. Same setup as evaluate.py."""
    hv = matrix_t[hist_idx]
    L = hv.shape[0]

    with torch.no_grad():
        fp, _ = model(hv.unsqueeze(0), torch.ones(1, L))
    r_model = topk_by_l2(fp.squeeze(0), matrix_t, paper_ids, K, exclude_set_idx=exclude)

    mp = F.normalize(hv.mean(dim=0), p=2, dim=-1)
    r_mp = topk_by_l2(mp, matrix_t, paper_ids, K, exclude_set_idx=exclude)

    return {
        'model_r': recall_at_k(r_model, positive, K),
        'model_n': ndcg_at_k(r_model, positive, K),
        'mp_r':    recall_at_k(r_mp, positive, K),
        'mp_n':    ndcg_at_k(r_mp, positive, K),
    }


def print_table(df, group_cols, title, note=None):
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    labels = "/".join(group_cols)
    hdr = (f"{labels:<16} {'n':>5} {'disp':>7} "
           f"{'model_R':>8} {'mp_R':>8} {'gap_R':>8} {'gap_R 95% CI':>18} "
           f"{'model_N':>8} {'mp_N':>8} {'gap_N':>8}")
    print(hdr)
    print("-" * len(hdr))
    for key, g in df.groupby(group_cols):
        key_s = "  ".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
        mr, pr = g['model_r'].mean(), g['mp_r'].mean()
        mn, pn = g['model_n'].mean(), g['mp_n'].mean()
        lo, hi = boot_gap_ci(g['model_r'].values, g['mp_r'].values)
        ci = f"[{lo:+.4f},{hi:+.4f}]"
        sig = "" if (lo <= 0 <= hi) else " *"
        # both methods pinned at zero recall -> the gap is a floor artifact, not a result
        floor = "  <- FLOOR: both ~0" if (mr < 1e-9 and pr < 1e-9) else ""
        print(f"{key_s:<16} {len(g):>5} {g['disp'].mean():>7.4f} "
              f"{mr:>8.4f} {pr:>8.4f} {mr - pr:>+8.4f} {ci:>18}{sig:<2} "
              f"{mn:>8.4f} {pn:>8.4f} {mn - pn:>+8.4f}{floor}")


def main():
    device = torch.device('cpu')
    rng = np.random.default_rng(SEED)

    raw = np.load(MATRIX_PATH).astype('float32')
    raw_t = torch.from_numpy(raw).to(device)
    matrix_t = F.normalize(raw_t, p=2, dim=1)
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    print(f"paper space: {tuple(matrix_t.shape)} (L2-normalized for retrieval)")

    model = UserHistoryEncoder(embedding_dim=EMBED_DIM).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    paper_field, field_papers, distant = build_field_maps()

    hist_df = pd.read_parquet(HISTORIES)
    test = hist_df[hist_df['split'] == 'test'].reset_index(drop=True)
    # shuffle once, then walk until we've collected N_HOMOG_USERS homogeneous ones
    test = test.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # --- select topically homogeneous users ---
    selected = []
    for _, row in test.iterrows():
        hist_ids = [str(x) for x in row['history_ids']]
        if any(h not in id_to_idx for h in hist_ids):
            continue
        if len(hist_ids) < max(K_LEVELS) + 1:   # need >=1 original paper left after max contamination
            continue
        fields = [paper_field.get(h) for h in hist_ids]
        if any(f is None for f in fields) or len(set(fields)) != 1:
            continue
        home = fields[0]
        if not distant.get(home):
            continue
        selected.append((row, hist_ids, home))
        if len(selected) >= N_HOMOG_USERS:
            break

    print(f"selected {len(selected)} topically homogeneous test users "
          f"(all history papers share one primary field)")
    home_counts = Counter(h for _, _, h in selected)
    print("home fields:", ", ".join(f"{f}={n}" for f, n in home_counts.most_common(6)))
    print("\ncontaminant pools (centroid cosine distance from home field):")
    for f, _n in home_counts.most_common(4):
        picks = distant[f]
        pool = sum(len(field_papers[p]) for p, _ in picks)
        desc = ", ".join(f"{p} (d={d:.3f})" for p, d in picks)
        print(f"  {f:<20} -> {desc}  [pool={pool} papers]")

    # --- run the k = 0, 1, 2 conditions ---
    rows = []
    for i, (row, hist_ids, home) in enumerate(selected):
        if i and i % 250 == 0:
            print(f"  ...{i}/{len(selected)}")

        positive = str(row['positive_id'])
        hist_idx = [id_to_idx[h] for h in hist_ids]
        L = len(hist_idx)

        # contaminant candidates: papers from the distant fields, minus anything the
        # user already has and minus the target itself
        pool = []
        for f, _d in distant[home]:
            pool.extend(field_papers[f])
        banned = set(hist_ids) | {positive}
        pool = [p for p in pool if p not in banned and p in id_to_idx]
        if len(pool) < max(K_LEVELS):
            continue

        # one fixed draw per user: which papers come in, which positions go out.
        # k=2 reuses k=1's swap, so the conditions nest.
        contam_ids = list(rng.choice(pool, size=max(K_LEVELS), replace=False))
        contam_idx = [id_to_idx[c] for c in contam_ids]
        positions = list(rng.choice(L, size=max(K_LEVELS), replace=False))

        # candidate pool is IDENTICAL across k: original history + every contaminant.
        # (the target is never in here -- it's excluded from both by construction)
        exclude = set(hist_idx) | set(contam_idx)

        for k in K_LEVELS:
            cur = list(hist_idx)
            for j in range(k):
                cur[positions[j]] = contam_idx[j]

            m = evaluate_history(model, cur, positive, exclude, matrix_t, paper_ids)

            # dispersion of the contaminated history, on raw embeddings -- confirms the
            # sabotage actually moved the history apart
            v = F.normalize(raw_t[cur], p=2, dim=1)
            sim = v @ v.T
            iu = torch.triu_indices(L, L, offset=1)
            disp = (1.0 - sim[iu[0], iu[1]]).mean().item()

            rows.append({'k': k, 'hist_len': L, 'home': home, 'disp': disp, **m})

    df = pd.DataFrame(rows)

    print("\n" + "=" * 92)
    print("CONTAMINATION: k history papers replaced with distant-field papers")
    print("gap = model - meanpool.  negative = model LOSES to mean-pool.")
    print("disp = mean pairwise cosine distance of the (contaminated) history.")
    print("=" * 92)

    print_table(df, ['k'], "[1] MAIN: metrics vs contamination level k",
                note="same users at every k; length and candidate pool held constant")

    print_table(df, ['hist_len', 'k'],
                "[2] BY HISTORY LENGTH (k=2 is a bigger dose of a 3-paper history than a 5-paper one)")

    # does the model's relative standing improve as the history gets dirtier?
    print("\ngap_R by k (the number that matters):")
    for k in K_LEVELS:
        g = df[df['k'] == k]
        lo, hi = boot_gap_ci(g['model_r'].values, g['mp_r'].values)
        print(f"  k={k}: gap_R={g['model_r'].mean() - g['mp_r'].mean():+.4f} "
              f"95%CI=[{lo:+.4f},{hi:+.4f}]  "
              f"absolute: model_R={g['model_r'].mean():.4f} mp_R={g['mp_r'].mean():.4f}  n={len(g)}")

    print("\nHow to read this: mean-pool CANNOT ignore a contaminant -- its centroid moves with "
          "every\njunk paper. If attention were gating, the model's gap_R should IMPROVE as k rises. "
          "A gap\nthat stays flat or worsens with k means the attention layer is not learning to "
          "down-weight\noff-topic history, which is an architecture finding, not a data one.")
    print("\nCRITICAL CAVEAT -- check the ABSOLUTE recalls before celebrating a shrinking gap.\n"
          "If model_R and mp_R both fall to ~0 as k rises, the gap closes only because BOTH\n"
          "methods were annihilated. That is a floor effect, not graceful degradation: you cannot\n"
          "distinguish 'attention gates well' from 'nothing retrieves anything' when both are zero.\n"
          "A shrinking gap is only evidence for gating if model_R holds up in absolute terms.\n")


if __name__ == "__main__":
    main()
