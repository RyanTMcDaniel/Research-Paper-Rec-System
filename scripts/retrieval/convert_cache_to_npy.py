"""
Step 1 of 2: convert the torch embedding cache into plain numpy arrays.

This script imports torch but NOT faiss. Keeping torch and faiss in separate
processes avoids the dual-OpenMP runtime collision that hard-crashes Python on
Apple Silicon. Run this once; then run build_faiss_index.py (faiss only).

Output:
  paper_matrix.npy  -- (N, 256) float32 matrix of embeddings
  paper_ids.npy     -- (N,) array of paperIds, aligned row-for-row with the matrix
"""

import os

import numpy as np
import torch

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE       = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
EMB_PATH   = os.path.join(BASE, "embedding_cache.pt")   # CHANGE for trained embeddings later
MATRIX_OUT = os.path.join(BASE, "paper_matrix.npy")
IDS_OUT    = os.path.join(BASE, "paper_ids.npy")
EMBED_DIM  = 256


def main():
    print(f"loading torch cache from {EMB_PATH} ...")
    cache = torch.load(EMB_PATH)
    paper_ids = list(cache.keys())

    # stack every vector into one contiguous matrix, then numpy float32 (FAISS requirement)
    matrix = torch.stack([cache[pid] for pid in paper_ids]).numpy().astype("float32")

    assert matrix.shape[1] == EMBED_DIM, f"expected dim {EMBED_DIM}, got {matrix.shape[1]}"
    assert not np.isnan(matrix).any(), "NaN in embeddings -- do not save a corrupt matrix"

    np.save(MATRIX_OUT, matrix)
    np.save(IDS_OUT, np.array(paper_ids))

    print(f"saved matrix -> {MATRIX_OUT}  shape {matrix.shape}")
    print(f"saved ids    -> {IDS_OUT}     count {len(paper_ids)}")


if __name__ == "__main__":
    main()