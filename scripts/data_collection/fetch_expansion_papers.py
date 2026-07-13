"""
Phase 3, Step 1 — Fetch metadata for out-of-corpus referenced papers.

Reads data/citations/citation_data.jsonl to find every referenced paperId
that is NOT already in data/cleaned_corpus.parquet, then batch-fetches
their full metadata via /paper/batch (same fields as the original corpus).

Papers cited by fewer than MIN_INCOMING_CITATIONS existing corpus papers
can be excluded to reduce fetch scope — raise this threshold (e.g. to 2)
if you want to skip low-value candidates.

Output: one JSONL file per 5,000 papers in data/raw_expansion/.
Checkpointed so it can resume after interruption.
"""

import json
import os
import time
from collections import defaultdict

import pandas as pd
import requests

# ---------- CONFIG ----------
CORPUS_PATH = "data/training_data/cleaned_corpus.parquet"
CITATIONS_PATH = "data/citations/citation_data.jsonl"
OUTPUT_DIR = "data/raw_expansion"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "fetch_checkpoint.json")

# Only fetch papers cited by at least this many existing in-corpus papers.
# Set to 1 to fetch everything. Raise to 2+ to skip low-density candidates.
MIN_INCOMING_CITATIONS = 1

BATCH_SIZE = 50         # 500-ID batches trigger 429 on full-fields requests; 50 is confirmed safe
FILE_BATCH_SIZE = 5_000  # papers per output JSONL file (matches original script)
REQUEST_DELAY = 1.1     # safely above 1 req/sec with API key

# Same fields as the original corpus pull.
FIELDS = "paperId,externalIds,title,year,citationCount,openAccessPdf,s2FieldsOfStudy,authors,abstract"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# -----------------------------

API_KEY = os.environ.get("S2_API_KEY")


def find_new_paper_ids(corpus_ids, citations_path, min_incoming):
    """Return sorted list of out-of-corpus referenced paperIds that meet the
    incoming-citation threshold."""
    incoming = defaultdict(int)
    with open(citations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for ref in record.get("references", []):
                if ref and ref not in corpus_ids:
                    incoming[ref] += 1

    candidates = [pid for pid, count in incoming.items() if count >= min_incoming]
    print(f"  Out-of-corpus referenced papers: {len(incoming):,}")
    print(f"  Meeting MIN_INCOMING_CITATIONS={min_incoming}: {len(candidates):,}")
    return sorted(candidates)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"chunk_index": 0, "papers_processed": 0, "file_batch_num": 0}


def save_checkpoint(chunk_index, papers_processed, file_batch_num):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(
            {"chunk_index": chunk_index,
             "papers_processed": papers_processed,
             "file_batch_num": file_batch_num},
            f,
        )


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


def flush_file_batch(current_batch, file_batch_num, output_dir):
    if not current_batch:
        return
    path = os.path.join(output_dir, f"papers_batch_{file_batch_num:04d}.jsonl")
    with open(path, "w") as f:
        for paper in current_batch:
            f.write(json.dumps(paper) + "\n")
    print(f"  Wrote {len(current_batch):,} papers to {path}")


def main():
    if API_KEY:
        print("API key loaded from environment.")
    else:
        print("WARNING: No API key found in S2_API_KEY. Running unauthenticated.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading corpus from {CORPUS_PATH}...")
    corpus_ids = set(
        pd.read_parquet(CORPUS_PATH, columns=["paperId"])["paperId"].dropna().tolist()
    )
    print(f"  Existing corpus: {len(corpus_ids):,} papers")

    print("Identifying out-of-corpus referenced papers...")
    new_ids = find_new_paper_ids(corpus_ids, CITATIONS_PATH, MIN_INCOMING_CITATIONS)
    total = len(new_ids)

    checkpoint = load_checkpoint()
    chunk_index = checkpoint["chunk_index"]
    papers_processed = checkpoint["papers_processed"]
    file_batch_num = checkpoint["file_batch_num"]

    chunks = [new_ids[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_chunks = len(chunks)
    print(f"Resuming from chunk {chunk_index}/{total_chunks} ({papers_processed:,}/{total:,} fetched)")

    current_file_batch = []

    try:
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
                if not paper.get("paperId"):
                    continue
                current_file_batch.append(paper)

                if len(current_file_batch) >= FILE_BATCH_SIZE:
                    flush_file_batch(current_file_batch, file_batch_num, OUTPUT_DIR)
                    current_file_batch = []
                    file_batch_num += 1

            papers_processed += len(chunk)
            chunk_index = ci + 1
            save_checkpoint(chunk_index, papers_processed, file_batch_num)

            if (ci + 1) % 10 == 0 or ci == total_chunks - 1:
                print(
                    f"  Chunk {ci + 1}/{total_chunks} | "
                    f"{papers_processed:,}/{total:,} processed"
                )

            if ci < total_chunks - 1:
                time.sleep(REQUEST_DELAY)

    finally:
        if current_file_batch:
            flush_file_batch(current_file_batch, file_batch_num, OUTPUT_DIR)
            file_batch_num += 1
            # Save checkpoint with incremented file_batch_num so a resume
            # won't overwrite the file we just wrote.
            save_checkpoint(chunk_index, papers_processed, file_batch_num)

    print(f"\nDone. Raw expansion papers in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
