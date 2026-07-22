"""
Build the co-trained corpus index.

Applies the CO-TRAINED paper projection (scripts/models/paper_projection_cotrained.pt) to the raw
768-d pooled SciBERT matrix paper_768_full.npy -> 253703 x 256, row order == paper_ids.npy.

Saved raw / un-normalized -- evaluate.py L2-normalizes at load (line 131), consistent with the
co-training, where the user tower's history input was normalize(proj(hist)). Query and index thus
live in one geometry.

Torch only, never FAISS. Refuses to overwrite an existing file.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

BASE       = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
P768       = os.path.join(BASE, "paper_768_full.npy")
IDS_PATH   = os.path.join(BASE, "paper_ids.npy")
OLD_MATRIX = os.path.join(BASE, "paper_matrix.npy")                 # random-proj R (sanity ref)
NEW_MATRIX = os.path.join(BASE, "paper_matrix_cotrained.npy")       # output
PROJ_CKPT  = os.path.join(REPO_ROOT, "scripts", "models", "paper_projection_cotrained.pt")

IN_DIM, OUT_DIM, BATCH = 768, 256, 8192


def main():
    if os.path.exists(NEW_MATRIX):
        sys.exit(f"{NEW_MATRIX} already exists -- refusing to overwrite. Delete it to rebuild.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    proj = nn.Linear(IN_DIM, OUT_DIM)
    proj.load_state_dict(torch.load(PROJ_CKPT, map_location="cpu"))
    proj.to(device).eval()
    for p in proj.parameters():
        p.requires_grad = False

    P = np.load(P768).astype("float32")
    ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    assert P.shape == (len(ids), IN_DIM), (P.shape, len(ids))

    out = np.empty((P.shape[0], OUT_DIM), dtype="float32")
    with torch.no_grad():
        for s in range(0, P.shape[0], BATCH):
            out[s:s + BATCH] = proj(torch.from_numpy(P[s:s + BATCH]).to(device)).cpu().numpy()
    assert not np.isnan(out).any(), "NaN in co-trained matrix -- refusing to save"

    np.save(NEW_MATRIX, out)
    print(f"saved -> {NEW_MATRIX}  shape {out.shape}  (row order == paper_ids.npy)")

    old = np.load(OLD_MATRIX).astype("float32")
    o = F.normalize(torch.from_numpy(old), p=2, dim=1)
    n = F.normalize(torch.from_numpy(out), p=2, dim=1)
    per_row = (o * n).sum(1)
    print(f"cosine(random-proj R, co-trained): mean={per_row.mean():.4f}  median={per_row.median():.4f}")
    print(f"norms: R mean={np.linalg.norm(old, axis=1).mean():.3f}  cotrained mean={np.linalg.norm(out, axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
