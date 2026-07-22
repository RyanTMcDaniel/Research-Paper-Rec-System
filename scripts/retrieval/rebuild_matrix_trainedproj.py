"""
Rebuild the corpus matrix with the TRAINED paper projection.

The existing paper_matrix.npy was produced by a PaperEncoder whose 768->256 projection was
random-init (build_embedding_cache.py never called load_state_dict). This re-encodes the same
253,703 papers with the trained paper_encoder.projection from two_tower_model_best.pt.

Tokenization and pooling are NOT reimplemented -- we call the same build_embedding_cache()
used originally, so the only thing that differs is the projection weights.

Row order is forced to match paper_ids.npy, because paper_matrix.npy is aligned row-for-row
with it (see convert_cache_to_npy.py) and every downstream id_to_idx lookup depends on that.

Writes paper_embeddings_trainedproj.npy. Does NOT overwrite the existing cache.
Torch only. FAISS is never imported.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from scripts.data_collection.build_embedding_cache import (  # noqa: E402
    build_embedding_cache,
    load_trained_paper_encoder,
)

BASE       = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
CORPUS     = os.path.join(BASE, "cleaned_corpus.parquet")
IDS_PATH   = os.path.join(BASE, "paper_ids.npy")
OLD_MATRIX = os.path.join(BASE, "paper_matrix.npy")
NEW_MATRIX = os.path.join(BASE, "paper_embeddings_trainedproj.npy")   # new file, old untouched

CKPT       = os.path.join(REPO_ROOT, "two_tower_model_best.pt")
MODEL_NAME = "allenai/scibert_scivocab_uncased"
BATCH_SIZE = 64
EMBED_DIM  = 256


def main():
    from transformers import AutoTokenizer

    if os.path.exists(NEW_MATRIX):
        sys.exit(f"{NEW_MATRIX} already exists -- refusing to overwrite. Delete it to rebuild.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    paper_encoder = load_trained_paper_encoder(CKPT, MODEL_NAME, device, output_dim=EMBED_DIM)

    # identical tokenization + mean-pooling to the original run; only the projection changed
    print("encoding 253,703 papers with the TRAINED projection (this is the slow part)...",
          flush=True)
    cache = build_embedding_cache(CORPUS, paper_encoder, tokenizer, device, batch_size=BATCH_SIZE)
    print(f"encoded {len(cache):,} papers", flush=True)

    # --- align to paper_ids.npy row order (NOT corpus order) ---
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    missing = [p for p in paper_ids if p not in cache]
    if missing:
        sys.exit(f"{len(missing)} ids in paper_ids.npy have no embedding -- aborting")

    new = torch.stack([cache[p] for p in paper_ids]).numpy().astype("float32")
    assert not np.isnan(new).any(), "NaN in new embeddings -- refusing to save"
    np.save(NEW_MATRIX, new)
    print(f"\nsaved -> {NEW_MATRIX}  shape {new.shape}", flush=True)

    # ---------------- SANITY CHECK ----------------
    old = np.load(OLD_MATRIX).astype("float32")
    print("\n--- sanity check: old (random proj) vs new (trained proj) ---")
    print(f"  old shape {old.shape}   new shape {new.shape}   match={old.shape == new.shape}")

    o = F.normalize(torch.from_numpy(old), p=2, dim=1)
    n = F.normalize(torch.from_numpy(new), p=2, dim=1)
    per_row = (o * n).sum(1)
    print(f"  per-paper cosine(old, new): mean={per_row.mean():.4f}  "
          f"median={per_row.median():.4f}  p1={per_row.kthvalue(max(1, len(per_row)//100)).values:.4f}  "
          f"max={per_row.max():.4f}")
    print(f"  vector norms: old mean={np.linalg.norm(old, axis=1).mean():.3f}  "
          f"new mean={np.linalg.norm(new, axis=1).mean():.3f}")
    print("  (near 0.0 => the trained projection really is a different map, confirming the bug;")
    print("   near 1.0 => the old cache was already the trained projection and nothing was wrong)")


if __name__ == "__main__":
    main()
