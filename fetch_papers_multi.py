"""
Semantic Scholar bulk paper fetcher — multi-query version.

Runs through a list of CS/ML/AI subtopic queries sequentially, filters each
page to CS-relevant results client-side (server-side fieldsOfStudy filtering
is unreliable — confirmed by testing), dedupes papers across queries by
paperId, writes results to local JSONL files in batches, and checkpoints
progress per-query so the script can resume after a crash without re-pulling
already-completed queries or already-seen papers.
"""

import requests
import json
import time
import os

# ---------- CONFIG ----------
QUERIES = [
    "deep learning",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "neural network",
    "recommender systems",
    "transformer model",
    "graph neural network",
    "machine learning",
    "artificial intelligence",
    "convolutional neural network",
    "generative adversarial network",
    "large language model",
    "transfer learning",
    "speech recognition",
    "robotics",
    "knowledge graph",
    "image classification",
    "object detection",
    "data mining",
    "anomaly detection",
    "time series forecasting",
    "federated learning",
    "self-supervised learning",
]
FIELDS = "paperId,title,abstract,authors,citationCount,year,s2FieldsOfStudy,externalIds"
TARGET_CATEGORIES = {"Computer Science", "Engineering", "Mathematics"}  # widened from CS-only
MAX_PAPERS = 500_000              # overall cap across all queries combined
MAX_PER_QUERY = 100_000           # cap per individual query, so one broad term can't eat the whole budget
BATCH_SIZE = 5_000                 # how many papers per output JSONL file
REQUEST_DELAY = 1.1                # stay above the 1 req/sec limit even with a key — don't drop this to 1.0
OUTPUT_DIR = "data/raw"
CHECKPOINT_FILE = "checkpoint.json"
SEEN_IDS_FILE = "seen_ids.txt"      # tracks paperIds already collected, across all queries

# API key is read from the S2_API_KEY environment variable — never hardcode it here.
# Set it in your terminal before running:
#   export S2_API_KEY="s2k-your-key-here"
API_KEY = os.environ.get("S2_API_KEY")
# -----------------------------

URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def load_checkpoint():
    """Resume from last saved state if a checkpoint exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"query_index": 0, "token": None, "collected": 0, "batch_num": 0}


def save_checkpoint(query_index, token, collected, batch_num):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(
            {
                "query_index": query_index,
                "token": token,
                "collected": collected,
                "batch_num": batch_num,
            },
            f,
        )


def load_seen_ids():
    """Load the set of paperIds already collected in previous runs."""
    if os.path.exists(SEEN_IDS_FILE):
        with open(SEEN_IDS_FILE, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def append_seen_id(paper_id, file_handle):
    file_handle.write(paper_id + "\n")


def is_target_category(paper, target_categories):
    """Client-side filter — server-side fieldsOfStudy param is unreliable.
    Accepts a paper if ANY of its tagged categories overlap with target_categories."""
    fos = paper.get("s2FieldsOfStudy") or []
    categories = {f["category"] for f in fos if "category" in f}
    return bool(categories & target_categories)


def fetch_page(query, token=None):
    params = {"query": query, "fields": FIELDS}
    if token:
        params["token"] = token

    headers = {"x-api-key": API_KEY} if API_KEY else {}

    for attempt in range(5):
        try:
            resp = requests.get(URL, params=params, headers=headers, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"  Request error ({e}), retrying in 5s...")
            time.sleep(5)
            continue

        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print("  Rate limited (429), backing off 5s...")
            time.sleep(5)
        else:
            print(f"  Unexpected status {resp.status_code}: {resp.text[:200]}")
            time.sleep(5)

    raise RuntimeError("Failed after 5 retries")


def main():
    if API_KEY:
        print("API key loaded from environment. Running authenticated.")
    else:
        print("WARNING: No API key found in S2_API_KEY environment variable.")
        print("Running unauthenticated — slower, shared rate limit.")
        print('Set it with: export S2_API_KEY="s2k-your-key-here"\n')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint = load_checkpoint()
    query_index = checkpoint["query_index"]
    token = checkpoint["token"]
    collected = checkpoint["collected"]
    batch_num = checkpoint["batch_num"]

    seen_ids = load_seen_ids()

    print(f"Resuming: query_index={query_index} ({QUERIES[query_index] if query_index < len(QUERIES) else 'DONE'}), "
          f"collected={collected}, batch={batch_num}, already-seen papers={len(seen_ids)}")

    current_batch = []
    seen_ids_file = open(SEEN_IDS_FILE, "a")

    def flush_batch():
        nonlocal current_batch, batch_num
        if not current_batch:
            return
        path = os.path.join(OUTPUT_DIR, f"papers_batch_{batch_num:04d}.jsonl")
        with open(path, "w") as f:
            for paper in current_batch:
                f.write(json.dumps(paper) + "\n")
        print(f"  Wrote {len(current_batch)} papers to {path}")
        current_batch = []
        batch_num += 1

    try:
        while query_index < len(QUERIES) and collected < MAX_PAPERS:
            query = QUERIES[query_index]
            query_collected = 0  # papers collected for THIS query specifically

            print(f"\n--- Query [{query_index + 1}/{len(QUERIES)}]: '{query}' ---")

            while query_collected < MAX_PER_QUERY and collected < MAX_PAPERS:
                data = fetch_page(query, token)
                page_results = data.get("data", [])

                if not page_results:
                    print("  No more results for this query.")
                    break

                for paper in page_results:
                    paper_id = paper.get("paperId")

                    if not paper_id or paper_id in seen_ids:
                        continue  # skip duplicates across queries

                    if is_target_category(paper, TARGET_CATEGORIES):
                        current_batch.append(paper)
                        seen_ids.add(paper_id)
                        append_seen_id(paper_id, seen_ids_file)
                        seen_ids_file.flush()

                        collected += 1
                        query_collected += 1

                        if len(current_batch) >= BATCH_SIZE:
                            flush_batch()

                        if query_collected >= MAX_PER_QUERY or collected >= MAX_PAPERS:
                            break

                token = data.get("token")
                save_checkpoint(query_index, token, collected, batch_num)

                print(f"  Progress: {collected}/{MAX_PAPERS} total | "
                      f"{query_collected}/{MAX_PER_QUERY} for this query | "
                      f"(scanned {data.get('total')} total matches for this query)")

                if not token:
                    print("  Reached end of result set for this query.")
                    break

                time.sleep(REQUEST_DELAY)

            # Move to next query, reset token
            query_index += 1
            token = None
            save_checkpoint(query_index, token, collected, batch_num)

    finally:
        flush_batch()  # always write any remaining partial batch, even on crash/interrupt
        seen_ids_file.close()

    print(f"\nDone. Total unique CS-relevant papers collected: {collected}")


if __name__ == "__main__":
    main()