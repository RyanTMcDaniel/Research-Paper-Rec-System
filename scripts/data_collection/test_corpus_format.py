import pandas as pd

# Load both
corpus = pd.read_parquet('data/training_data/cleaned_corpus.parquet')
histories = pd.read_parquet('data/training_data/histories.parquet')

# --- 1. ID type/format match ---
print("=== ID TYPES ===")
print("corpus paperId dtype:", corpus['paperId'].dtype)
print("corpus paperId sample:", corpus['paperId'].iloc[:3].tolist())

print("\nhistories positive_id dtype:", histories['positive_id'].dtype)
print("histories positive_id sample:", histories['positive_id'].iloc[:3].tolist())

# history_ids is presumably a list column — check the inner type
sample_hist_ids = histories['history_ids'].iloc[0]
print("\nhistory_ids sample row:", sample_hist_ids, "| inner type:", type(sample_hist_ids[0]))

# --- 2. Actual overlap check (the real test, not just "do they look similar") ---
corpus_ids = set(corpus['paperId'])

positive_ids = set(histories['positive_id'])
missing_positive = positive_ids - corpus_ids
print(f"\npositive_id: {len(positive_ids)} unique, {len(missing_positive)} NOT found in corpus")

all_history_ids = set(hid for row in histories['history_ids'] for hid in row)
missing_history = all_history_ids - corpus_ids
print(f"history_ids: {len(all_history_ids)} unique, {len(missing_history)} NOT found in corpus")

if missing_positive:
    print("example missing positive_id:", list(missing_positive)[:3])
if missing_history:
    print("example missing history_id:", list(missing_history)[:3])

# --- 3. Null abstract check ---
print("\n=== ABSTRACT NULLS ===")
n_null = corpus['abstract'].isna().sum()
print(f"{n_null} / {len(corpus)} papers ({100*n_null/len(corpus):.1f}%) have null abstract")