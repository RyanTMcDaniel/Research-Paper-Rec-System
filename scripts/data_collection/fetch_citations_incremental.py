"""
Phase 3, Step 3 — Fetch citation/reference data for newly added papers only.

Reads the current cleaned_corpus.parquet (after expansion), compares against
paperIds already present in data/citations/citation_data.jsonl, and fetches
reference data only for the delta (papers in the corpus but not yet in the
JSONL). Appends results to the existing JSONL so the full file covers the
entire expanded corpus for build_histories.py.

Run this after clean_expansion.py has updated the corpus. After it completes,
re-run build_histories.py to regenerate training data with the denser graph.
"""

import json
import os
import time

import pandas as pd
import requests

# ---------- CONFIG ----------
CORPUS_PATH = "data/cleaned_corpus.parquet"
CITATIONS_PATH = "data/citations/citation_data.jsonl"
CHECKPOINT_FILE = "data/citations/fetch_incremental_checkpoint.json"

BATCH_SIZE = 500
REQUEST_DELAY = 1.1
FIELDS = "references.paperId,year"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# -----------------------------

API_KEY = os.environ.get("S2_API_KEY")


def already_fetched_ids(citations_path):
    """Scan the existing JSONL to find paperIds we already have citation data for."""
    seen = set()
    if not os.path.exists(citations_path):
        return seen
    with open(citations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pid = record.get("paperId")
                if pid:
                    seen.add(pid)
            except json.JSONDecodeError:
                continue
    return seen


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"chunk_index": 0, "papers_processed": 0}


def save_checkpoint(chunk_index, papers_processed):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"chunk_index": chunk_index, "papers_processed": papers_processed}, f)


def fetch_batch(paper_ids):
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    params = {"fields": FIELDS}
    body = {"ids": paper_ids}

    for attempt in range(5):
        try:
            resp = requests.post(
                BATCH_URL, params=params, headers=headers, json=body, timeout=30
            )
        except requests.exceptions.RequestException as e:
            print(f"  Request error ({e}), retrying in 5s...")
            time.sleep(5)
            continue

        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s, 480s
            print(f"  Rate limited (429), backing off {wait}s...")
            time.sleep(wait)
        else:
            print(f"  Unexpected status {resp.status_code}: {resp.text[:200]}")
            time.sleep(5)

    raise RuntimeError("Failed after 5 retries")


def main():
    if API_KEY:
        print("API key loaded from environment.")
    else:
        print("WARNING: No API key found in S2_API_KEY. Running unauthenticated.")

    print(f"Loading corpus from {CORPUS_PATH}...")
    all_ids = set(
        pd.read_parquet(CORPUS_PATH, columns=["paperId"])["paperId"].dropna().tolist()
    )
    print(f"  Total corpus: {len(all_ids):,} papers")

    print(f"Scanning {CITATIONS_PATH} for already-fetched IDs...")
    fetched = already_fetched_ids(CITATIONS_PATH)
    print(f"  Already fetched: {len(fetched):,}")

    delta = sorted(all_ids - fetched)
    total = len(delta)
    print(f"  To fetch (delta): {total:,}")

    if total == 0:
        print("Nothing to fetch — citation data is already complete for the full corpus.")
        return

    checkpoint = load_checkpoint()
    chunk_index = checkpoint["chunk_index"]
    papers_processed = checkpoint["papers_processed"]

    chunks = [delta[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_chunks = len(chunks)
    print(
        f"Resuming from chunk {chunk_index}/{total_chunks} "
        f"({papers_processed:,}/{total:,} new papers fetched)"
    )
    print(
        f"Estimated time: {(total_chunks - chunk_index) * REQUEST_DELAY / 60:.1f} minutes"
    )

    # Append to existing JSONL
    with open(CITATIONS_PATH, "a") as out_f:
        for ci in range(chunk_index, total_chunks):
            chunk = chunks[ci]

            try:
                results = fetch_batch(chunk)
            except RuntimeError as e:
                print(f"  Fatal error on chunk {ci}: {e}. Progress saved, exiting.")
                break

            for paper in results:
                if paper is None:
                    continue
                paper_id = paper.get("paperId")
                if not paper_id:
                    continue

                references = [
                    ref["paperId"]
                    for ref in (paper.get("references") or [])
                    if ref and ref.get("paperId")
                ]
                record = {
                    "paperId": paper_id,
                    "year": paper.get("year"),
                    "references": references,
                }
                out_f.write(json.dumps(record) + "\n")

            papers_processed += len(chunk)
            chunk_index = ci + 1
            save_checkpoint(chunk_index, papers_processed)

            if (ci + 1) % 10 == 0 or ci == total_chunks - 1:
                print(
                    f"  Chunk {ci + 1}/{total_chunks} | "
                    f"{papers_processed:,}/{total:,} new papers processed"
                )

            if ci < total_chunks - 1:
                time.sleep(REQUEST_DELAY)

    print(f"\nDone. {CITATIONS_PATH} now covers the full expanded corpus.")
    print("Next step: python3 scripts/training_data/build_histories.py")


if __name__ == "__main__":
    main()
