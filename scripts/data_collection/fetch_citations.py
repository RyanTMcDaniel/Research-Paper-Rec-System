"""
Phase 2, Step 1 — Fetch citation/reference data for all in-corpus papers.

For each paper in cleaned_corpus.parquet, fetches its reference list (papers
it cites) and its own year via the /paper/batch endpoint (up to 500 IDs per
request). Writes one JSONL record per paper to data/citations/citation_data.jsonl.

We only need references (not citations) because build_histories.py reconstructs
the co-citation graph from an inverted index over the reference lists.

Checkpoints progress so the script can resume after interruption. Same
pattern as fetch_papers_multi.py: chunk index → JSONL, retry on failure,
1.1s delay between requests.
"""

import json
import os
import time

import pandas as pd
import requests

# ---------- CONFIG ----------
CORPUS_PATH = "data/cleaned_corpus.parquet"
OUTPUT_DIR = "data/citations"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "citation_data.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "fetch_checkpoint.json")

BATCH_SIZE = 500        # /paper/batch hard limit
REQUEST_DELAY = 1.1     # safely above 1 req/sec with API key
FIELDS = "references.paperId,year"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# -----------------------------

# Read from environment — never hardcode.
API_KEY = os.environ.get("S2_API_KEY")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"chunk_index": 0, "papers_processed": 0}


def save_checkpoint(chunk_index, papers_processed):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"chunk_index": chunk_index, "papers_processed": papers_processed}, f)


def fetch_batch(paper_ids):
    """POST up to 500 IDs to /paper/batch. Returns list of paper dicts (null for not-found IDs)."""
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
            print("  Rate limited (429), backing off 10s...")
            time.sleep(10)
        else:
            print(f"  Unexpected status {resp.status_code}: {resp.text[:200]}")
            time.sleep(5)

    raise RuntimeError("Failed after 5 retries")


def main():
    if API_KEY:
        print("API key loaded from environment.")
    else:
        print("WARNING: No API key found in S2_API_KEY. Running unauthenticated — slower rate limits.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading corpus from {CORPUS_PATH}...")
    df = pd.read_parquet(CORPUS_PATH, columns=["paperId"])
    all_paper_ids = df["paperId"].dropna().tolist()
    total = len(all_paper_ids)
    print(f"Total in-corpus papers to process: {total:,}")

    checkpoint = load_checkpoint()
    chunk_index = checkpoint["chunk_index"]
    papers_processed = checkpoint["papers_processed"]

    chunks = [all_paper_ids[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_chunks = len(chunks)
    print(f"Resuming from chunk {chunk_index}/{total_chunks} ({papers_processed:,}/{total:,} processed)")

    # Append mode: already-processed chunks are already in the file.
    with open(OUTPUT_PATH, "a") as out_f:
        for ci in range(chunk_index, total_chunks):
            chunk = chunks[ci]

            try:
                results = fetch_batch(chunk)
            except RuntimeError as e:
                print(f"  Fatal error on chunk {ci}: {e}. Progress saved, exiting.")
                break

            for paper in results:
                # Batch endpoint returns null for paperIds it can't resolve.
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
                    f"{papers_processed:,}/{total:,} papers processed"
                )

            if ci < total_chunks - 1:
                time.sleep(REQUEST_DELAY)

    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
