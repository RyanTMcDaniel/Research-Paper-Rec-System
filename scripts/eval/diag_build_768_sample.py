"""
STAGE 1 of the embedding-space diagnosis. Builds the one artifact the other
script needs and cannot get from disk: RAW 768-d mean-pooled SciBERT vectors.

Why this has to exist. embedding_cache.pt is already 256-d -- centroid_calc.py
says so at line 30 ("confirmed: pooled to 256, NOT 768"). The pre-projection
768-d space was never saved. So H1 ("the projection destroyed the structure")
is untestable until we re-run the frozen SciBERT backbone and mean-pool it
ourselves. That is inference with a frozen encoder, not retraining.

Fidelity to the original pipeline matters here, because stage 2 fits
cached_256 ~ affine(pooled_768) and reads the residual. So this script copies
build_embedding_cache.py EXACTLY: same model, same title+abstract text field,
same truncation at 512, same attention-mask mean-pool. The only thing it does
NOT do is apply PaperEncoder.projection -- that's the layer under investigation.

Sampling: every category is capped at CAT_CAP papers so the 23 smaller fields
are actually represented, plus a uniform random slice for the anisotropy tests.
Writes {paperId: float32[768]} to the scratch dir. Torch only, no FAISS.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluate import BASE, CORPUS, IDS_PATH  # noqa: E402  (reuse the existing paths)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "retrieval"))
from centroid_calc import get_cats  # noqa: E402  (reuse the EXACT category parser)

SCRATCH   = "/private/tmp/claude-501/-Users-ryanmcdaniel/a7fd4cc6-2a49-4466-80b7-a29a78771a2e/scratchpad"
OUT_PATH  = os.path.join(SCRATCH, "sample_768.pt")

MODEL_NAME = "allenai/scibert_scivocab_uncased"   # same backbone as build_embedding_cache.py
CAT_CAP    = 600     # per-category cap so tail fields (History=445) are fully covered
N_RANDOM   = 3000    # extra uniform sample, for the H2 anisotropy / H4 metric tests
MAX_LEN    = 512     # MUST match build_embedding_cache.py or the affine fit is invalid
BATCH_SIZE = 32
SEED       = 42


def mean_pool(hidden_states, attention_mask):
    """Attention-masked mean pool -- byte-for-byte the same op as PaperEncoder.forward."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def main():
    from transformers import AutoModel, AutoTokenizer

    rng = np.random.default_rng(SEED)
    os.makedirs(SCRATCH, exist_ok=True)

    print("[1/5] loading corpus...", flush=True)
    # only the columns we need -- the abstract column makes a full load slow and fat
    corpus = pd.read_parquet(CORPUS, columns=["paperId", "title", "abstract", "s2FieldsOfStudy"])
    print(f"      {len(corpus):,} rows", flush=True)

    print("[2/5] filtering to indexed papers + parsing categories...", flush=True)
    indexed = set(str(p) for p in np.load(IDS_PATH, allow_pickle=True))
    corpus = corpus[corpus["paperId"].astype(str).isin(indexed)].reset_index(drop=True)
    corpus["cat_list"] = corpus["s2FieldsOfStudy"].apply(get_cats)   # reused parser
    print(f"      {len(corpus):,} indexed papers", flush=True)

    # --- stratified sample: cap every category, so tail fields aren't drowned by CS ---
    exploded = (corpus[["paperId", "cat_list"]]
                .explode("cat_list").dropna(subset=["cat_list"])
                .rename(columns={"cat_list": "category"}))

    print("[3/5] stratified sampling...", flush=True)
    picked = set()
    for cat, grp in exploded.groupby("category"):
        ids = grp["paperId"].astype(str).unique()
        take = ids if len(ids) <= CAT_CAP else rng.choice(ids, size=CAT_CAP, replace=False)
        picked.update(take.tolist() if hasattr(take, "tolist") else list(take))

    all_ids = corpus["paperId"].astype(str).to_numpy()
    picked.update(rng.choice(all_ids, size=min(N_RANDOM, len(all_ids)), replace=False).tolist())

    sample = corpus[corpus["paperId"].astype(str).isin(picked)].reset_index(drop=True)
    print(f"      {len(sample):,} papers "
          f"(<= {CAT_CAP}/category across {exploded['category'].nunique()} categories + {N_RANDOM} random)",
          flush=True)

    # --- same text field as build_embedding_cache.py ---
    sample["text"] = sample["title"].fillna("") + " " + sample["abstract"].fillna("")

    print("[4/5] loading SciBERT...", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in bert.parameters():
        p.requires_grad = False
    print(f"      device={device}", flush=True)

    print(f"[5/5] embedding {len(sample):,} papers...", flush=True)
    t0 = time.time()
    cache = {}
    n_batches = (len(sample) + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.no_grad():
        for bi, start in enumerate(range(0, len(sample), BATCH_SIZE)):
            b = sample.iloc[start:start + BATCH_SIZE]
            enc = tok(b["text"].tolist(), padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            hs = bert(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"]).last_hidden_state
            pooled = mean_pool(hs, enc["attention_mask"]).cpu().numpy().astype(np.float32)
            for pid, v in zip(b["paperId"].astype(str).tolist(), pooled):
                cache[pid] = v
            if bi % 20 == 0:
                done = start + len(b)
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(sample) - done) / max(rate, 1e-9)
                print(f"      {done:>6}/{len(sample)}  {rate:5.1f} papers/s  eta {eta/60:5.1f} min",
                      flush=True)

    torch.save(cache, OUT_PATH)
    d = next(iter(cache.values())).shape[0]
    print(f"saved {len(cache):,} raw pooled vectors (dim {d}) -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
