"""
Phase 2, Step 2 — Build synthetic user reading histories from citation data.

Reads cleaned_corpus.parquet and data/citations/citation_data.jsonl
(produced by fetch_citations.py) to construct synthetic reading sessions
for two-tower triplet-loss training.

Two chain signals
-----------------
1. Reference-list chains: in-corpus paper C's in-corpus references form a
   "reading session" — the author of C likely read those papers together.
   Chain seed = C.  Chain year = C's publication year.

2. Bibliographic coupling chains: multiple in-corpus papers that all cite
   the same reference R form a chain — they independently built on the same
   prior work.  Chain seed = R's paperId (prefixed "bib_").  Chain year =
   median year of papers in the chain.

Sliding windows (history len 3–5)
----------------------------------
For each sorted chain of length N, extract every contiguous window of size
W in {4, 5, 6} (= history_len + 1), yielding:
  history_ids  = window[:-1]   (list of 3–5 in-corpus paperIds)
  positive_id  = window[-1]    (held-out next paper)

Papers with fewer than MIN_SEED_CONNECTIONS in-corpus neighbours are not
used as chain seeds (they can still appear as positives or negatives).

Splitting
---------
Assigned at the CHAIN level (not example level) to prevent overlapping
windows from the same chain leaking across train/val/test.  Split boundary
is temporal by chain_year.  Chains without a resolvable year go to train.

Output: data/training_data/histories.parquet
----------------------------------------------
Columns:
  chain_id, chain_type, chain_year       — chain provenance
  history_ids (list[str]), positive_id   — the training signal
  history_len, positive_year             — for DataLoader / analysis
  positive_primary_field                 — tag for cross-domain eval
  split                                  — train / val / test

Negatives are NOT pre-sampled here.  Sample them in the PyTorch DataLoader
(random or hard-negative from FAISS) so they stay fresh across epochs.
"""

import json
import os
import statistics

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict

# ---------- CONFIG ----------
CORPUS_PATH = "data/training_data/cleaned_corpus.parquet"
CITATIONS_PATH = "data/citations/citation_data.jsonl"
OUTPUT_DIR = "data/training_data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "histories.parquet")

MIN_HISTORY_LEN = 3
MAX_HISTORY_LEN = 5
# Minimum in-corpus neighbours for a paper to be used as a chain seed.
# = MIN_HISTORY_LEN + 1 so we can form at least one (history, positive) pair.
MIN_SEED_CONNECTIONS = MIN_HISTORY_LEN + 1  # 4

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# TEST_FRAC = remaining 0.10
# -----------------------------


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path):
    df = pd.read_parquet(path, columns=["paperId", "year", "citationCount", "s2FieldsOfStudy"])
    df = df.dropna(subset=["paperId"])
    return df


def _parse_primary_field(s2_fields):
    """Return the first s2FieldsOfStudy category string, or None."""
    if s2_fields is None:
        return None
    if isinstance(s2_fields, float):  # NaN from pandas
        return None
    if isinstance(s2_fields, str):
        try:
            s2_fields = json.loads(s2_fields)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(s2_fields, (list, tuple)) or not s2_fields:
        return None
    first = s2_fields[0]
    return first.get("category") if isinstance(first, dict) else None


def build_corpus_lookups(df):
    """Return (corpus_ids, paper_year, paper_cites, paper_primary_field)."""
    corpus_ids = set(df["paperId"].tolist())

    paper_year = {}
    paper_cites = {}
    paper_primary_field = {}

    for row in df.itertuples(index=False):
        pid = row.paperId
        paper_year[pid] = int(row.year) if pd.notna(row.year) else None
        paper_cites[pid] = int(row.citationCount) if pd.notna(row.citationCount) else 0
        paper_primary_field[pid] = _parse_primary_field(
            getattr(row, "s2FieldsOfStudy", None)
        )

    return corpus_ids, paper_year, paper_cites, paper_primary_field


# ---------------------------------------------------------------------------
# Citation data loading
# ---------------------------------------------------------------------------

def load_citation_data(citations_path, corpus_ids):
    """
    Returns paper_refs: {in-corpus paperId -> list of in-corpus reference paperIds}.
    Also supplements paper_year with years from the API response (preferred over
    corpus metadata when both are available, since the API year is authoritative).
    """
    paper_refs = {}
    api_year = {}

    with open(citations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            pid = record.get("paperId")
            if not pid or pid not in corpus_ids:
                continue

            in_corpus_refs = [r for r in record.get("references", []) if r in corpus_ids]
            paper_refs[pid] = in_corpus_refs

            yr = record.get("year")
            if yr is not None:
                api_year[pid] = int(yr)

    return paper_refs, api_year


# ---------------------------------------------------------------------------
# Chain construction helpers
# ---------------------------------------------------------------------------

def sort_chain(paper_ids, paper_year, paper_cites):
    """Sort by year asc; break ties by citationCount desc (more cited = more foundational)."""
    def key(pid):
        return (paper_year.get(pid) or 0, -(paper_cites.get(pid) or 0))
    return sorted(paper_ids, key=key)


def extract_windows(chain, chain_id, chain_type, chain_year, paper_year, paper_primary_field):
    """
    Yield all sliding windows of total length W in [MIN_HISTORY_LEN+1, MAX_HISTORY_LEN+1].
    Each window produces one (history_ids, positive_id) example.
    """
    n = len(chain)
    examples = []

    for total_len in range(MIN_HISTORY_LEN + 1, MAX_HISTORY_LEN + 2):
        history_len = total_len - 1
        for i in range(n - total_len + 1):
            window = chain[i : i + total_len]
            positive_id = window[-1]
            examples.append({
                "chain_id": chain_id,
                "chain_type": chain_type,
                "chain_year": chain_year,
                "history_ids": list(window[:-1]),
                "positive_id": positive_id,
                "history_len": history_len,
                "positive_year": paper_year.get(positive_id),
                "positive_primary_field": paper_primary_field.get(positive_id),
                "history_primary_fields": [paper_primary_field.get(pid) for pid in window[:-1]],
            })

    return examples


# ---------------------------------------------------------------------------
# Temporal split assignment (at chain level)
# ---------------------------------------------------------------------------

def assign_splits(examples, train_frac, val_frac):
    """
    All examples from the same chain go to the same split.
    Chains are ranked by chain_year; cutoffs are set at train_frac and
    train_frac+val_frac quantiles of the year-sorted chain list.
    Chains with no resolvable year are assigned to train (conservative).
    """
    # One entry per chain
    chain_year_map = {}
    for ex in examples:
        cid = ex["chain_id"]
        if cid not in chain_year_map:
            chain_year_map[cid] = ex.get("chain_year")

    # Sort chains that have a year
    chains_with_year = sorted(
        [(cid, yr) for cid, yr in chain_year_map.items() if yr is not None],
        key=lambda x: x[1],
    )
    n = len(chains_with_year)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    chain_to_split = {}
    for i, (cid, _) in enumerate(chains_with_year):
        if i < train_end:
            chain_to_split[cid] = "train"
        elif i < val_end:
            chain_to_split[cid] = "val"
        else:
            chain_to_split[cid] = "test"

    for cid in chain_year_map:
        chain_to_split.setdefault(cid, "train")  # no year → train

    for ex in examples:
        ex["split"] = chain_to_split[ex["chain_id"]]

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading corpus from {CORPUS_PATH}...")
    corpus_df = load_corpus(CORPUS_PATH)
    corpus_ids, paper_year, paper_cites, paper_primary_field = build_corpus_lookups(corpus_df)
    print(f"Corpus: {len(corpus_ids):,} papers")

    print(f"Loading citation data from {CITATIONS_PATH}...")
    paper_refs, api_year = load_citation_data(CITATIONS_PATH, corpus_ids)
    # API year is more authoritative; supplement (not override) corpus year
    for pid, yr in api_year.items():
        if pid not in paper_year or paper_year[pid] is None:
            paper_year[pid] = yr
    print(f"Citation data loaded for {len(paper_refs):,} papers")

    all_examples = []

    # ------------------------------------------------------------------
    # Signal 1: reference-list chains
    # Each in-corpus paper C seeds a chain from its in-corpus references.
    # ------------------------------------------------------------------
    print("\nBuilding reference-list chains...")
    ref_chains = 0
    ref_skipped_sparse = 0
    for paper_id, refs in paper_refs.items():
        if len(refs) < MIN_SEED_CONNECTIONS:
            ref_skipped_sparse += 1
            continue

        chain = sort_chain(refs, paper_year, paper_cites)
        chain_year = paper_year.get(paper_id)
        examples = extract_windows(
            chain, paper_id, "reference", chain_year, paper_year, paper_primary_field
        )
        all_examples.extend(examples)
        ref_chains += 1

    print(f"  Seeds used: {ref_chains:,} | skipped (sparse): {ref_skipped_sparse:,}")
    print(f"  Examples so far: {len(all_examples):,}")

    # ------------------------------------------------------------------
    # Signal 2: bibliographic coupling chains
    # Papers that all cite the same reference R form a chain.
    # Build inverted index from reference lists.
    # ------------------------------------------------------------------
    print("Building bibliographic coupling chains...")
    ref_to_citers = defaultdict(list)
    for paper_id, refs in paper_refs.items():
        for ref_id in refs:
            ref_to_citers[ref_id].append(paper_id)

    bib_chains = 0
    bib_skipped_sparse = 0
    for ref_id, citers in ref_to_citers.items():
        if len(citers) < MIN_SEED_CONNECTIONS:
            bib_skipped_sparse += 1
            continue

        chain = sort_chain(citers, paper_year, paper_cites)
        years = [paper_year.get(pid) for pid in chain if paper_year.get(pid) is not None]
        chain_year = int(statistics.median(years)) if years else None

        examples = extract_windows(
            chain, f"bib_{ref_id}", "bibcoupling", chain_year, paper_year, paper_primary_field
        )
        all_examples.extend(examples)
        bib_chains += 1

    print(f"  Seeds used: {bib_chains:,} | skipped (sparse): {bib_skipped_sparse:,}")
    print(f"  Total examples before split: {len(all_examples):,}")

    # ------------------------------------------------------------------
    # Assign train/val/test splits at chain level
    # ------------------------------------------------------------------
    print("\nAssigning temporal splits at chain level...")
    all_examples = assign_splits(all_examples, TRAIN_FRAC, VAL_FRAC)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for ex in all_examples:
        split_counts[ex["split"]] += 1
    for split, count in split_counts.items():
        pct = count / len(all_examples) * 100
        print(f"  {split}: {count:,} ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # Save to Parquet
    # ------------------------------------------------------------------
    print(f"\nSaving to {OUTPUT_PATH}...")

    schema = pa.schema([
        pa.field("chain_id", pa.string()),
        pa.field("chain_type", pa.string()),
        pa.field("chain_year", pa.int32()),
        pa.field("history_ids", pa.list_(pa.string())),
        pa.field("positive_id", pa.string()),
        pa.field("history_len", pa.int8()),
        pa.field("positive_year", pa.int32()),
        pa.field("positive_primary_field", pa.string()),
        pa.field("history_primary_fields", pa.list_(pa.string())),
        pa.field("split", pa.string()),
    ])

    table = pa.Table.from_pydict(
        {
            "chain_id":               [ex["chain_id"] for ex in all_examples],
            "chain_type":             [ex["chain_type"] for ex in all_examples],
            "chain_year":             [ex.get("chain_year") for ex in all_examples],
            "history_ids":            [ex["history_ids"] for ex in all_examples],
            "positive_id":            [ex["positive_id"] for ex in all_examples],
            "history_len":            [ex["history_len"] for ex in all_examples],
            "positive_year":          [ex.get("positive_year") for ex in all_examples],
            "positive_primary_field":  [ex.get("positive_primary_field") for ex in all_examples],
            "history_primary_fields":  [ex["history_primary_fields"] for ex in all_examples],
            "split":                   [ex["split"] for ex in all_examples],
        },
        schema=schema,
    )

    pq.write_table(table, OUTPUT_PATH)
    print(f"Saved {len(all_examples):,} examples to {OUTPUT_PATH}")
    print(f"\nQuick load:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_parquet('{OUTPUT_PATH}')")
    print(f"  df.groupby('split').size()")


if __name__ == "__main__":
    main()
