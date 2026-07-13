"""
Assert that no co-citation chain spans train/val/test splits.

build_histories.py assigns splits at the chain level, so 0 spanning chains
should hold by construction — this script verifies it against the shipped
artifact instead of trusting the construction.
"""
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES = os.path.join(BASE, "histories.parquet")

df = pd.read_parquet(HISTORIES, columns=["chain_id", "split"])
splits_per_chain = df.groupby("chain_id")["split"].nunique()
spanning = (splits_per_chain > 1).sum()

print(f"examples: {len(df):,}")
print(f"chains:   {len(splits_per_chain):,}")
print(f"chains spanning more than one split: {spanning}")

assert spanning == 0, f"{spanning} chains leak across splits"
print("PASS: 0 chains span splits")
