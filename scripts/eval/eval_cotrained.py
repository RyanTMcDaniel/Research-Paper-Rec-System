"""
CAPSTONE eval: co-trained model vs mean-pool on the full 253,703-paper index.

SINGLE geometry (query == index by construction, unlike the decoupled prior experiment):
  index  = paper_matrix_cotrained.npy  (co-trained projection of all papers)
  query  = cotrained_user_tower( normalize(index[history]) )
Both live in the one co-trained space. This is exactly evaluate.py's native single-matrix path,
so we reuse its functions verbatim (imported, not modified) and point them at the co-trained
matrix + co-trained user tower. Same 2000 test users (random_state=42) as the baseline.

The ONLY judge of success is this full-index R@10 / NDCG@10 -- not the training/val triplet loss
(semi-hard negatives make that loss an unreliable proxy by design).

Torch only, never FAISS. CPU (mirrors evaluate.py).
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
C_MATRIX   = os.path.join(BASE, "paper_matrix_cotrained.npy")
IDS_PATH   = os.path.join(BASE, "paper_ids.npy")
USER_CKPT  = os.path.join(REPO_ROOT, "scripts", "models", "user_tower_cotrained.pt")

EMBED_DIM, K, N_EVAL_USERS, SEED = 256, 10, 2000, 42

# prior rows (this project), for the comparison table
ROW_BASELINE = ("user-tower-only (baseline)", 0.0140, 0.0074, 0.0225, 0.0113)
ROW_PAPERONLY = ("paper-proj-only (decoupled)", 0.0090, 0.0047, 0.0300, 0.0152)


def main():
    device = torch.device("cpu")

    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    C = F.normalize(torch.from_numpy(np.load(C_MATRIX).astype("float32")), p=2, dim=1)
    print(f"co-trained index C {tuple(C.shape)} (L2-normalized)")

    model = UserHistoryEncoder(embedding_dim=EMBED_DIM).to(device)
    model.load_state_dict(torch.load(USER_CKPT, map_location=device))
    model.eval()

    histories_df = pd.read_parquet(HISTORIES)
    print("\n>>> space check on the co-trained index (norms must be ~1.0):")
    space_check(model, C, id_to_idx, paper_ids, histories_df)

    test_df = histories_df[histories_df['split'] == 'test'].reset_index(drop=True)
    test_df = test_df.sample(n=min(N_EVAL_USERS, len(test_df)), random_state=SEED).reset_index(drop=True)
    print(f"evaluating {len(test_df)} test users (same set as baseline)\n")

    sc = {k: {'recall': [], 'ndcg': []} for k in ('model', 'meanpool')}

    def add(key, ranked, positive):
        sc[key]['recall'].append(recall_at_k(ranked, positive, K))
        sc[key]['ndcg'].append(ndcg_at_k(ranked, positive, K))

    for _, row in test_df.iterrows():
        positive = str(row['positive_id'])
        hist_idx = [id_to_idx[str(h)] for h in row['history_ids'] if str(h) in id_to_idx]
        if not hist_idx:
            continue
        exclude = set(hist_idx)
        H = C[hist_idx]                                     # co-trained history vectors (normalized)

        with torch.no_grad():                              # query in the co-trained geometry
            fp, _ = model(H.unsqueeze(0), torch.ones(1, H.shape[0]))
        add('model', topk_by_l2(fp.squeeze(0), C, paper_ids, K, exclude), positive)
        add('meanpool', topk_by_l2(F.normalize(H.mean(0), p=2, dim=-1), C, paper_ids, K, exclude), positive)

    mr, mn = float(np.mean(sc['model']['recall'])), float(np.mean(sc['model']['ndcg']))
    pr, pn = float(np.mean(sc['meanpool']['recall'])), float(np.mean(sc['meanpool']['ndcg']))

    print("Query and index BOTH live in the co-trained geometry (single-geometry eval).\n")
    hdr = f"{'configuration':<30}{'model R@10':>12}{'model NDCG':>12}{'mp R@10':>12}{'mp NDCG':>12}{'gap(R@10)':>12}"
    print(hdr); print("-" * len(hdr))
    for label, r, n, pr0, pn0 in (ROW_BASELINE, ROW_PAPERONLY):
        print(f"{label:<30}{r:>12.4f}{n:>12.4f}{pr0:>12.4f}{pn0:>12.4f}{r - pr0:>+12.4f}")
    print(f"{'co-trained (both towers)':<30}{mr:>12.4f}{mn:>12.4f}{pr:>12.4f}{pn:>12.4f}{mr - pr:>+12.4f}")

    beat = mr > pr
    print(f"\nco-trained: model R@10={mr:.4f}  mean-pool R@10={pr:.4f}  gap={mr - pr:+.4f}")
    print(f"VERDICT: co-training {'LET the model BEAT mean-pool' if beat else 'did NOT let the model beat mean-pool'} "
          f"(model {'>' if beat else '<='} mean-pool on R@10).")


if __name__ == "__main__":
    main()
