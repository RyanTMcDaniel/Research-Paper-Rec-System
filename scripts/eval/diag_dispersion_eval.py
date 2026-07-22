"""
Dispersion-stratified eval: does attention only earn its keep when the history is
topically DIVERSE?

Hypothesis. Mean-pool is a perfectly good summarizer of a tight, single-topic
history -- the centroid IS the user. Attention can only add value when the
history is spread out and some items deserve more weight than others. If the
model's deficit vs mean-pool shrinks (or flips) as history dispersion rises,
the architecture isn't broken, it's just being asked to reweight histories that
have nothing worth reweighting.

Dispersion score = mean pairwise cosine distance between the user's RAW history
embeddings (raw = straight out of paper_matrix.npy, pre-normalization). Cosine is
scale-invariant, so this is the same number you'd get on the unit sphere; we read
it off the raw vectors because that's the space the histories actually live in.

Two tables:
  1. Dispersion quartiles over all test users.
  2. The same quartile breakdown restricted to history_len == 4, which de-confounds
     diversity from length -- longer histories are mechanically more dispersed, so
     without this you can't tell which variable is doing the work.

TORCH ONLY -- no faiss. Reuses evaluate.py's model class, normalized geometry,
raw-space paper index, metrics, and checkpoint by direct import, so the numbers
here are apples-to-apples with the headline eval.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the EXACT eval setup -- same model def, same metrics, same search, same paths.
from evaluate import (  # noqa: E402
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

N_EVAL_USERS = 12000  # Recall@10 here is ~1-2%, so a bucket of n=300 holds only ~5 hits.
                      # 12k users -> ~1000/bucket in the histlen==4 table, which is the
                      # bare minimum for the gap CIs below to say anything.
FOCUS_LEN    = 4      # the length we re-run the quartile split inside
N_BOOT       = 2000   # paired bootstrap resamples for the gap CI
SEED         = 42


def boot_gap_ci(model_hits, mp_hits, n_boot=N_BOOT, seed=SEED):
    """
    95% CI on the PAIRED gap (model - meanpool), resampling users with replacement.
    Paired because both methods are scored on the same users -- that cancels the
    per-user difficulty and is the only reason a gap this small is estimable at all.
    Returns (lo, hi). If the interval straddles 0, the gap is not distinguishable
    from noise, no matter how suggestive the point estimate looks.
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


def dispersion(raw_vecs):
    """
    Mean pairwise cosine DISTANCE (1 - cos sim) over the raw history embeddings.
    0.0 = every paper points the same way (perfectly homogeneous history).
    Higher = the history spans more of the space.
    """
    v = F.normalize(raw_vecs, p=2, dim=1)          # cosine == dot on unit vectors
    sim = v @ v.T                                  # (L, L)
    L = v.shape[0]
    iu = torch.triu_indices(L, L, offset=1)        # upper triangle == the L*(L-1)/2 pairs
    return (1.0 - sim[iu[0], iu[1]]).mean().item()


def print_quartile_table(df, title):
    """Quartile-bucket `df` on its dispersion column and print model vs mean-pool."""
    print(f"\n{title}")
    print(f"  users: {len(df)}")
    if len(df) < 8:
        print("  too few users to quartile -- skipping")
        return

    # qcut on this subset's own distribution: quartile edges are recomputed per table,
    # which is the whole point of the histlen==4 rerun.
    try:
        q = pd.qcut(df['disp'], 4, labels=['Q1 (tightest)', 'Q2', 'Q3', 'Q4 (most diverse)'],
                    duplicates='drop')
    except ValueError:
        print("  dispersion has too little spread to form quartiles -- skipping")
        return
    df = df.assign(bucket=q)

    hdr = (f"{'bucket':<20} {'n':>5} {'disp_rng':>13} "
           f"{'model_R':>8} {'mp_R':>8} {'gap_R':>8} {'gap_R 95% CI':>18} "
           f"{'model_N':>8} {'mp_N':>8} {'gap_N':>8}")
    print(hdr)
    print("-" * len(hdr))

    def row(label, g, disp_rng=""):
        mr, pr = g['model_r'].mean(), g['mp_r'].mean()
        mn, pn = g['model_n'].mean(), g['mp_n'].mean()
        lo, hi = boot_gap_ci(g['model_r'].values, g['mp_r'].values)
        ci = f"[{lo:+.4f},{hi:+.4f}]"
        sig = "" if (lo <= 0 <= hi) else " *"   # * = CI excludes 0
        print(f"{label:<20} {len(g):>5} {disp_rng:>13} "
              f"{mr:>8.4f} {pr:>8.4f} {mr - pr:>+8.4f} {ci:>18}{sig:<2} "
              f"{mn:>8.4f} {pn:>8.4f} {mn - pn:>+8.4f}")

    for b in df['bucket'].cat.categories:
        g = df[df['bucket'] == b]
        if len(g) == 0:
            continue
        row(b, g, f"{g['disp'].min():.3f}-{g['disp'].max():.3f}")

    print("-" * len(hdr))
    row('ALL', df)


def main():
    device = torch.device('cpu')

    # --- paper space: keep BOTH views ---
    # raw  -> dispersion is defined on the raw history embeddings
    # norm -> retrieval must happen on the unit sphere the model was trained on
    raw = np.load(MATRIX_PATH).astype('float32')
    raw_t = torch.from_numpy(raw).to(device)
    matrix_t = F.normalize(raw_t, p=2, dim=1)
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    print(f"paper space: {tuple(matrix_t.shape)} (raw kept for dispersion, L2-normalized for retrieval)")

    model = UserHistoryEncoder(embedding_dim=EMBED_DIM).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    hist_df = pd.read_parquet(HISTORIES)
    test = hist_df[hist_df['split'] == 'test'].reset_index(drop=True)
    if N_EVAL_USERS:
        test = test.sample(n=min(N_EVAL_USERS, len(test)), random_state=SEED).reset_index(drop=True)
    print(f"evaluating {len(test)} test users\n")

    rows = []
    for i, row in test.iterrows():
        if i and i % 500 == 0:
            print(f"  ...{i}/{len(test)}")
        hist_ids = [str(x) for x in row['history_ids']]
        positive = str(row['positive_id'])
        hist_idx = [id_to_idx[x] for x in hist_ids if x in id_to_idx]
        if len(hist_idx) < 2:            # need >=2 papers for a pairwise dispersion
            continue
        exclude = set(hist_idx)
        hv = matrix_t[hist_idx]          # normalized -> model + retrieval
        L = hv.shape[0]                  # usable length (after dropping ids not in the index)

        disp = dispersion(raw_t[hist_idx])   # raw -> dispersion

        with torch.no_grad():
            fp, _ = model(hv.unsqueeze(0), torch.ones(1, L))
        r_model = topk_by_l2(fp.squeeze(0), matrix_t, paper_ids, K, exclude_set_idx=exclude)

        mp = F.normalize(hv.mean(dim=0), p=2, dim=-1)
        r_mp = topk_by_l2(mp, matrix_t, paper_ids, K, exclude_set_idx=exclude)

        rows.append({
            'hist_len': L,
            'disp':     disp,
            'model_r':  recall_at_k(r_model, positive, K),
            'model_n':  ndcg_at_k(r_model, positive, K),
            'mp_r':     recall_at_k(r_mp, positive, K),
            'mp_n':     ndcg_at_k(r_mp, positive, K),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 96)
    print("DISPERSION = mean pairwise cosine distance between raw history embeddings")
    print("gap = model - meanpool.  negative = model LOSES to mean-pool.")
    print("=" * 96)

    print(f"\ndispersion distribution: mean={df['disp'].mean():.4f} "
          f"min={df['disp'].min():.4f} max={df['disp'].max():.4f}")
    print("mean dispersion by history_len (the confound we're removing):")
    for L, g in df.groupby('hist_len'):
        print(f"  len {L}: n={len(g):>5}  mean_disp={g['disp'].mean():.4f}")

    print_quartile_table(df, "[1] ALL TEST USERS, by dispersion quartile")

    sub = df[df['hist_len'] == FOCUS_LEN].copy()
    print_quartile_table(
        sub, f"[2] history_len == {FOCUS_LEN} ONLY, by dispersion quartile "
             f"(length held constant -> diversity is the only moving part)")

    print("\nHow to read this: if gap_R climbs toward 0 (or turns positive) from Q1 to Q4 "
          f"in table [2],\ndiversity -- not length -- is what attention needs. If the gap is flat "
          "across quartiles,\ndispersion isn't the explanation and the deficit is architectural.")
    print("\nBUT read the CIs first. Recall@10 is ~1-2% here, so a bucket holds only a handful of\n"
          "actual hits. Any gap whose 95% CI straddles 0 (no '*') is noise -- do not read a trend\n"
          "into a row of overlapping intervals.\n")


if __name__ == "__main__":
    main()
