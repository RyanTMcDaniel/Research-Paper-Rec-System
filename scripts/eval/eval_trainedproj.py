"""
Decoupled eval (Option A) for the trained paper projection.

The frozen user tower was trained on the random-projection space R (paper_matrix.npy) and is
in-distribution ONLY on R. So the model's query is built EXACTLY as in the baseline --
user_tower(R history) -- and is byte-for-byte the same for both rows below. The single thing
that changes between rows is the SEARCH INDEX:

    baseline row : query = user_tower(R hist)  ->  search R  (paper_matrix.npy)
    trained  row : query = user_tower(R hist)  ->  search T  (paper_matrix_trainedproj.npy)

The mean-pool baseline is the parameter-free control WITHIN each row's own space (average that
row's index vectors, search that index). Both rows are computed on the SAME 2000 test users so
the comparison is apples-to-apples.

Reuses evaluate.py verbatim (imported, not modified): UserHistoryEncoder, topk_by_l2,
recall_at_k, ndcg_at_k, space_check. Torch only, never FAISS. CPU (mirrors evaluate.py).
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from scripts.eval.evaluate import (  # noqa: E402
    UserHistoryEncoder, topk_by_l2, recall_at_k, ndcg_at_k, space_check,
)

BASE       = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES  = os.path.join(BASE, "histories.parquet")
R_MATRIX   = os.path.join(BASE, "paper_matrix.npy")                 # random-proj R
T_MATRIX   = os.path.join(BASE, "paper_matrix_trainedproj.npy")     # trained-proj T
IDS_PATH   = os.path.join(BASE, "paper_ids.npy")
USER_CKPT  = os.path.join(REPO_ROOT, "scripts", "models", "user_history_encoder_best.pt")

EMBED_DIM, K, N_EVAL_USERS, SEED = 256, 10, 2000, 42
# documented baseline for reference (README)
DOC_MODEL_R, DOC_MP_R = 0.0140, 0.0225


def main():
    device = torch.device("cpu")

    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    R = F.normalize(torch.from_numpy(np.load(R_MATRIX).astype("float32")), p=2, dim=1)
    T = F.normalize(torch.from_numpy(np.load(T_MATRIX).astype("float32")), p=2, dim=1)
    print(f"R (random-proj) {tuple(R.shape)}   T (trained-proj) {tuple(T.shape)}  -- both L2-normalized")

    model = UserHistoryEncoder(embedding_dim=EMBED_DIM).to(device)
    model.load_state_dict(torch.load(USER_CKPT, map_location=device))
    model.eval()

    histories_df = pd.read_parquet(HISTORIES)
    print("\n>>> space check on the TRAINED index T (norms must be ~1.0):")
    space_check(model, T, id_to_idx, paper_ids, histories_df)

    test_df = histories_df[histories_df['split'] == 'test'].reset_index(drop=True)
    test_df = test_df.sample(n=min(N_EVAL_USERS, len(test_df)), random_state=SEED).reset_index(drop=True)
    print(f"evaluating {len(test_df)} test users (same set for both rows)\n")

    keys = ['model_R', 'mp_R', 'model_T', 'mp_T']
    sc = {k: {'recall': [], 'ndcg': []} for k in keys}

    def add(key, ranked, positive):
        sc[key]['recall'].append(recall_at_k(ranked, positive, K))
        sc[key]['ndcg'].append(ndcg_at_k(ranked, positive, K))

    for _, row in test_df.iterrows():
        hist_ids = [str(h) for h in row['history_ids']]
        positive = str(row['positive_id'])
        hist_idx = [id_to_idx[h] for h in hist_ids if h in id_to_idx]
        if not hist_idx:
            continue
        exclude = set(hist_idx)

        # MODEL query -- built from R history, identical for both rows (Option A)
        R_hist = R[hist_idx]
        with torch.no_grad():
            fp, _ = model(R_hist.unsqueeze(0), torch.ones(1, R_hist.shape[0]))
        fp = fp.squeeze(0)

        # baseline row: search R
        add('model_R', topk_by_l2(fp, R, paper_ids, K, exclude), positive)
        add('mp_R', topk_by_l2(F.normalize(R_hist.mean(0), p=2, dim=-1), R, paper_ids, K, exclude), positive)

        # trained row: SAME query fp, search T; mean-pool control lives in T
        add('model_T', topk_by_l2(fp, T, paper_ids, K, exclude), positive)
        T_hist = T[hist_idx]
        add('mp_T', topk_by_l2(F.normalize(T_hist.mean(0), p=2, dim=-1), T, paper_ids, K, exclude), positive)

    def m(key, metric):
        return float(np.mean(sc[key][metric]))

    print("Model history INPUT stays R for BOTH rows (frozen tower is in-distribution only on R);")
    print("only the SEARCH INDEX changes (R -> T). This isolates the paper projection as the one variable.\n")

    hdr = f"{'projection':<22}{'model R@10':>12}{'model NDCG':>12}{'mp R@10':>12}{'mp NDCG':>12}{'gap(R@10)':>12}"
    print(hdr); print("-" * len(hdr))
    rows = [
        ("random (baseline)", 'model_R', 'mp_R'),
        ("trained",          'model_T', 'mp_T'),
    ]
    for label, mk, pk in rows:
        mr, mn, pr, pn = m(mk, 'recall'), m(mk, 'ndcg'), m(pk, 'recall'), m(pk, 'ndcg')
        print(f"{label:<22}{mr:>12.4f}{mn:>12.4f}{pr:>12.4f}{pn:>12.4f}{mr - pr:>+12.4f}")
    print(f"\n(reference: documented baseline model R@10={DOC_MODEL_R}, mean-pool R@10={DOC_MP_R})")

    base_gap = m('model_R', 'recall') - m('mp_R', 'recall')
    new_gap  = m('model_T', 'recall') - m('mp_T', 'recall')
    delta = new_gap - base_gap
    verdict = ("CLOSED" if delta > 1e-4 else "WIDENED" if delta < -1e-4 else "LEFT UNCHANGED")
    print(f"\nmodel-meanpool R@10 gap: baseline {base_gap:+.4f} -> trained {new_gap:+.4f}  (delta {delta:+.4f})")
    print(f"VERDICT: training the paper projection {verdict} the model-vs-meanpool gap.")
    print("Note: a trained space can lift mean-pool too, so the GAP (not the raw model number) is the verdict.")


if __name__ == "__main__":
    main()
