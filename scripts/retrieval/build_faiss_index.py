"""
Step 2 of 2: build a FAISS flat L2 index and verify nearest-neighbor search.

This script imports faiss + numpy ONLY -- no torch. That's deliberate: torch and
faiss each bundle their own OpenMP runtime, and loading both in one process hard-
crashes Python on Apple Silicon. Run convert_cache_to_npy.py first to produce the
.npy inputs this script reads.

Flat L2 (IndexFlatL2) is exact brute-force search and matches the p=2 Euclidean
distance the triplet loss trains on. Per the plan: start flat for correctness,
upgrade to IVF for speed later (Step 5 / Step 9).

Today: run against the FROZEN cache's numpy export to prove the pipeline works.
Tomorrow: re-export from the TRAINED paper tower, rerun this unchanged.
"""

import os
import time
import numpy as np
import faiss


# ---------- CONFIG ----------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE        = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
MATRIX_PATH = os.path.join(BASE, "paper_matrix.npy")
IDS_PATH    = os.path.join(BASE, "paper_ids.npy")
INDEX_OUT   = os.path.join(BASE, "paper_index.faiss")
IDMAP_OUT   = os.path.join(BASE, "paper_index_ids.npy")
EMBED_DIM   = 256


def load_embeddings(matrix_path, ids_path):
    """
    Load the precomputed numpy matrix and its aligned paperId array.
    matrix[i] is the embedding for paper_ids[i] -- that alignment is how a FAISS
    search result (a row number) maps back to a real paperId.
    """
    matrix = np.load(matrix_path)
    paper_ids = np.load(ids_path, allow_pickle=True)

    matrix = matrix.astype("float32")  # FAISS requires float32
    assert matrix.shape[1] == EMBED_DIM, f"expected dim {EMBED_DIM}, got {matrix.shape[1]}"
    assert not np.isnan(matrix).any(), "NaN in embeddings -- do not index a corrupt matrix"
    assert matrix.shape[0] == len(paper_ids), "matrix rows and id count must match"

    return matrix, paper_ids


def build_index(matrix):
    """Flat L2 index. No training step needed for a flat index -- just add vectors."""
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(matrix)
    return index


def sanity_check(index, matrix, paper_ids, k=5):
    """
    Query the index with a vector that IS in the index (row 0).
    A correct index must return row 0 itself as the #1 nearest neighbor with
    distance ~0. If it doesn't, the index is wired wrong.
    """
    query = matrix[0:1]  # (1, D) -- FAISS expects a 2D batch of queries
    distances, indices = index.search(query, k)

    print("\n--- sanity check: querying with paper row 0 ---")
    print("query paperId:", paper_ids[0])
    print("top-k result rows:   ", indices[0])
    print("top-k distances:     ", distances[0])
    print("top-k paperIds:      ", [paper_ids[i] for i in indices[0]])

    assert indices[0][0] == 0, "self should be the nearest neighbor -- index is wrong"
    assert distances[0][0] < 1e-3, f"self-distance should be ~0, got {distances[0][0]}"
    print("PASS: nearest neighbor of row 0 is itself at distance ~0")


def main():
    print(f"loading embeddings from {MATRIX_PATH} ...")
    matrix, paper_ids = load_embeddings(MATRIX_PATH, IDS_PATH)
    print(f"loaded {matrix.shape[0]} vectors, dim {matrix.shape[1]}")

    print("building flat L2 index ...")
    t0 = time.time()
    index = build_index(matrix)
    print(f"index built in {time.time() - t0:.2f}s, ntotal = {index.ntotal}")

    # timing: this is the sub-50ms retrieval number the plan targets
    q = matrix[0:1]
    t0 = time.time()
    index.search(q, 10)
    print(f"single query (k=10) took {(time.time() - t0) * 1000:.2f} ms")

    sanity_check(index, matrix, paper_ids)

    faiss.write_index(index, INDEX_OUT)
    np.save(IDMAP_OUT, np.array(paper_ids))
    print(f"\nsaved index -> {INDEX_OUT}")
    print(f"saved id map -> {IDMAP_OUT}")


if __name__ == "__main__":
    main()