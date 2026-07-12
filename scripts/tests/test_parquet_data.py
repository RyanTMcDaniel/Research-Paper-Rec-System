import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))

corpus = pd.read_parquet(os.path.join(BASE, "cleaned_corpus.parquet"))

col = corpus["s2FieldsOfStudy"]
print("total rows:", len(col))
print("null count:", col.isnull().sum())
print("non-null count:", col.notnull().sum())

# show the first few NON-null values, if any exist
non_null = col[col.notnull()]
print("\nfirst non-null values:")
for v in non_null.head(3):
    print("  type:", type(v), "| repr:", repr(v)[:150])