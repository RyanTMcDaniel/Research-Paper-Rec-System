"""
Phase 3, Step 2 — Clean expansion papers and merge into the corpus.

Reads raw JSONL files from data/raw_expansion/, applies the same cleaning
pipeline as clean_data.py:
  - CS/Engineering/Math field filter (client-side, same categories as original)
  - Drop missing or too-short abstracts
  - Dedupe by paperId against the existing corpus AND within the new batch
  - English-only filter

Survivors are appended to data/cleaned_corpus.parquet (existing rows are
never re-cleaned or re-written). The original file is backed up first.
"""

import glob
import json
import os

import pandas as pd
from langdetect import detect, LangDetectException

# ---------- CONFIG ----------
RAW_EXPANSION_DIR = "data/raw_expansion"
CORPUS_PATH = "data/cleaned_corpus.parquet"
BACKUP_PATH = "data/cleaned_corpus_pre_expansion.parquet"

TARGET_CATEGORIES = {"Computer Science", "Engineering", "Mathematics"}
MIN_ABSTRACT_LENGTH = 20  # characters
# -----------------------------


def load_expansion_batches(raw_dir):
    papers = []
    files = sorted(glob.glob(os.path.join(raw_dir, "papers_batch_*.jsonl")))
    print(f"Found {len(files)} expansion batch files")
    for filepath in files:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    papers.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  Skipping malformed line in {filepath}")
    print(f"Loaded {len(papers):,} raw expansion records")
    return papers


def is_target_category(paper):
    fos = paper.get("s2FieldsOfStudy") or []
    categories = {f["category"] for f in fos if isinstance(f, dict) and "category" in f}
    return bool(categories & TARGET_CATEGORIES)


def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_expansion(papers, existing_ids):
    df = pd.DataFrame(papers)
    start_count = len(df)
    print(f"\nStarting count: {start_count:,}")

    # Step A: CS/Engineering/Math field filter
    df = df[df.apply(is_target_category, axis=1)]
    print(f"After field filter: {len(df):,} (-{start_count - len(df):,})")
    step_count = len(df)

    # Step B: drop missing/short abstracts
    df = df[df["abstract"].notna()]
    df = df[df["abstract"].str.strip().str.len() >= MIN_ABSTRACT_LENGTH]
    print(f"After abstract filter: {len(df):,} (-{step_count - len(df):,})")
    step_count = len(df)

    # Step C: dedupe within the new batch
    df = df.drop_duplicates(subset="paperId", keep="first")
    print(f"After intra-batch dedupe: {len(df):,} (-{step_count - len(df):,})")
    step_count = len(df)

    # Step D: dedupe against existing corpus
    df = df[~df["paperId"].isin(existing_ids)]
    print(f"After corpus dedupe: {len(df):,} (-{step_count - len(df):,})")
    step_count = len(df)

    # Step E: English-only filter
    print("Running language detection...")
    sample_text = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    df = df[sample_text.apply(is_english)]
    print(f"After English filter: {len(df):,} (-{step_count - len(df):,})")

    retained_pct = len(df) / start_count * 100 if start_count else 0
    print(f"\nNew papers surviving cleaning: {len(df):,} / {start_count:,} ({retained_pct:.1f}%)")
    return df


def main():
    print(f"Loading existing corpus from {CORPUS_PATH}...")
    existing_df = pd.read_parquet(CORPUS_PATH)
    existing_ids = set(existing_df["paperId"].dropna().tolist())
    print(f"  Existing corpus: {len(existing_ids):,} papers")

    papers = load_expansion_batches(RAW_EXPANSION_DIR)
    if not papers:
        print("No expansion batch files found. Run fetch_expansion_papers.py first.")
        return

    new_df = clean_expansion(papers, existing_ids)
    if new_df.empty:
        print("No new papers survived cleaning.")
        return

    # Back up before modifying
    if not os.path.exists(BACKUP_PATH):
        existing_df.to_parquet(BACKUP_PATH, index=False)
        print(f"\nBacked up original corpus to {BACKUP_PATH}")

    # Align columns: new papers get the same columns as the existing corpus
    # (some API fields may differ; fill any missing columns with None)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_parquet(CORPUS_PATH, index=False)

    print(f"\nUpdated corpus: {len(existing_df):,} → {len(combined):,} papers (+{len(new_df):,})")
    print(f"Saved to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
