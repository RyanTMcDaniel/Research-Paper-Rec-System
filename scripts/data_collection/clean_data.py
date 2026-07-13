"""
Phase 1, Step 2 — Clean and filter the raw paper corpus.

Reads all data/raw/papers_batch_*.jsonl files, drops papers with missing
abstracts, filters to English-language papers, deduplicates by paperId
(belt-and-suspenders — should already be deduped from collection, but
verify rather than assume), and writes the cleaned corpus to a single
Parquet file for fast loading in later embedding steps.
"""

import json
import glob
import os
import pandas as pd
from langdetect import detect, LangDetectException

RAW_DIR = "data/raw"
OUTPUT_PARQUET = "data/training_data/cleaned_corpus.parquet"
MIN_ABSTRACT_LENGTH = 20  # characters — guards against junk like "N/A" or single words


def load_all_batches():
    """Read every JSONL batch file into a single list of paper dicts."""
    papers = []
    files = sorted(glob.glob(f"{RAW_DIR}/papers_batch_*.jsonl"))
    print(f"Found {len(files)} batch files")

    for filepath in files:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    papers.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  Skipping malformed line in {filepath}")
                    continue

    print(f"Loaded {len(papers)} raw paper records")
    return papers


def is_english(text):
    """Best-effort English detection. Treat detection failures as non-English (drop)."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_papers(papers):
    df = pd.DataFrame(papers)
    start_count = len(df)
    print(f"\nStarting count: {start_count}")

    # Step A: drop missing/empty/too-short abstracts
    df = df[df["abstract"].notna()]
    df = df[df["abstract"].str.strip().str.len() >= MIN_ABSTRACT_LENGTH]
    print(f"After dropping missing/short abstracts: {len(df)} (-{start_count - len(df)})")
    step_count = len(df)

    # Step B: dedupe by paperId (verification pass — should already be deduped at collection time)
    df = df.drop_duplicates(subset="paperId", keep="first")
    print(f"After deduplication: {len(df)} (-{step_count - len(df)})")
    step_count = len(df)

    # Step C: filter to English only
    # Run language detection on title + abstract combined for better signal on short text
    print("Running language detection (this takes a while for 100K+ rows)...")
    sample_text = (df["title"].fillna("") + ". " + df["abstract"].fillna(""))
    df = df[sample_text.apply(is_english)]
    print(f"After English-only filter: {len(df)} (-{step_count - len(df)})")

    print(f"\nFinal cleaned count: {len(df)} / {start_count} ({len(df)/start_count*100:.1f}% retained)")
    return df


def main():
    papers = load_all_batches()
    df = clean_papers(papers)

    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nWrote cleaned corpus to {OUTPUT_PARQUET}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()