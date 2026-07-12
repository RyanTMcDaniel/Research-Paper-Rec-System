"""
Kaggle: compute per-category centroids from a paper-embedding cache (.pt) +
a corpus parquet that holds the category labels. Saves to parquet WITH
per-category paper counts.

Data reality (confirmed by inspection):
  - embedding_cache.pt : dict {paperId(str) -> torch.Tensor[256]}, 253,703 entries
  - cleaned_corpus.pq  : has 'paperId' (same 40-char hex ids) and
                         's2FieldsOfStudy' = list[{'category': str, 'source': str}]
  - categories join to embeddings on paperId DIRECTLY. No hashing.
  - 23 categories after .title() casing-normalization (merges the
    'and'/'And' Agricultural split). Smallest real category ~445 papers.

Run once on Kaggle. Load the output parquet at eval time. Do NOT recompute.

>>> CHECK THE CONFIG BLOCK — CORPUS_PATH is the one thing I'm still guessing. <<<
"""

import numpy as np
import pandas as pd
import torch

# =============================================================================
# CONFIG
# =============================================================================
CACHE_PATH   = "/kaggle/input/datasets/ryanthomasmcdaniel/embedding-cache/embedding_cache.pt"
CORPUS_PATH  = "/kaggle/input/datasets/ryanthomasmcdaniel/new-rec-data/cleaned_corpus.parquet"   
PAPER_ID_COL = "paperId"                                # id column in the corpus
FOS_COL      = "s2FieldsOfStudy"                         # list-of-dicts category column
EMBEDDING_DIM = 256                                     # confirmed: pooled to 256, NOT 768
OUTPUT_PATH  = "/kaggle/working/category_centroids.parquet"
MIN_CAT_SIZE = 50                                        # drop categories thinner than this
                                                        # (kills the 2-paper ghost, keeps History@445)
# =============================================================================


def load_cache():
    """embedding_cache.pt -> dict {paperId: np.ndarray[256] float32}."""
    obj = torch.load(CACHE_PATH, map_location="cpu")
    assert isinstance(obj, dict), f"expected dict cache, got {type(obj)}"
    cache = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in obj.items()}
    sample = next(iter(cache.values()))
    assert sample.shape == (EMBEDDING_DIM,), \
        f"expected dim {EMBEDDING_DIM}, got {sample.shape} — fix EMBEDDING_DIM"
    print(f"cache: {len(cache):,} papers, dim {sample.shape[0]}")
    return cache


def clean_cat(s):
    """Casing-only normalization. Hex inspection proved the only dupe was
    a capital-A 'And' vs lowercase 'and' — no hidden characters."""
    return s.strip().title()


def get_cats(cell):
    """Pull normalized category strings out of the s2FieldsOfStudy list-of-dicts."""
    if cell is None:
        return []
    out = []
    for d in cell:
        if isinstance(d, dict) and d.get("category"):
            out.append(clean_cat(d["category"]))
    return out


def build_pairs(corpus, cache):
    """
    One (paperId, category) row per tag, keeping only papers whose embedding
    is actually in the cache and categories that clear MIN_CAT_SIZE.
    """
    corpus = corpus[[PAPER_ID_COL, FOS_COL]].copy()
    corpus["cat_list"] = corpus[FOS_COL].apply(get_cats)

    have_emb = corpus[PAPER_ID_COL].isin(cache.keys())
    dropped = (~have_emb).sum()
    if dropped:
        print(f"note: {dropped:,} corpus papers have no cached embedding — skipped")
    corpus = corpus[have_emb]

    exploded = corpus[[PAPER_ID_COL, "cat_list"]].explode("cat_list")
    exploded = exploded.dropna(subset=["cat_list"])
    exploded = exploded.rename(columns={"cat_list": "category"})

    counts = exploded["category"].value_counts()
    valid = set(counts[counts >= MIN_CAT_SIZE].index)
    before = exploded["category"].nunique()
    exploded = exploded[exploded["category"].isin(valid)]
    after = exploded["category"].nunique()
    print(f"categories: {before} -> {after} after MIN_CAT_SIZE={MIN_CAT_SIZE} floor")
    return exploded


def compute_centroids(pairs, cache):
    rows = []
    for category, grp in pairs.groupby("category"):
        ids = grp[PAPER_ID_COL].to_numpy()
        vecs = np.stack([cache[pid] for pid in ids])
        rows.append({
            "category": category,
            "n_papers": int(len(ids)),
            "centroid": vecs.mean(axis=0).astype(np.float32),
        })
    return pd.DataFrame(rows).sort_values("n_papers", ascending=False).reset_index(drop=True)


def main():
    cache = load_cache()
    corpus = pd.read_parquet(CORPUS_PATH)
    print(f"corpus: {len(corpus):,} rows, id+fos present: "
          f"{PAPER_ID_COL in corpus.columns and FOS_COL in corpus.columns}")

    pairs = build_pairs(corpus, cache)
    centroids = compute_centroids(pairs, cache)

    print(f"\n=== {len(centroids)} category centroids ===")
    print(centroids[["category", "n_papers"]].to_string(index=False))

    overall = np.stack(list(cache.values())).mean(axis=0)
    cs_row = centroids[centroids["category"] == "Computer Science"]
    if len(cs_row):
        cs_vec = cs_row.iloc[0]["centroid"]
        cos = float(np.dot(cs_vec, overall) /
                    (np.linalg.norm(cs_vec) * np.linalg.norm(overall) + 1e-9))
        print(f"\nCS-centroid vs whole-corpus-mean cosine: {cos:.4f} "
              f"(near 1.0 = CS is a weak signal, as expected)")

    cmat = np.stack(centroids["centroid"].to_numpy())
    dim_df = pd.DataFrame(cmat, columns=[f"dim_{i}" for i in range(cmat.shape[1])])
    final = pd.concat(
        [centroids[["category", "n_papers"]].reset_index(drop=True), dim_df],
        axis=1,
    )
    final.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nsaved {len(final)} centroids -> {OUTPUT_PATH}")
    print(f"shape {final.shape}  (category + n_papers + {cmat.shape[1]} dims)")


if __name__ == "__main__":
    main()