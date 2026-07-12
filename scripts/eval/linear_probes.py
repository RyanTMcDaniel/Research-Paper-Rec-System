"""
Linear probe: can a linear classifier predict field-of-study from frozen
paper embeddings? High accuracy => embeddings linearly encode real semantic
structure. Chance-level => the space is noise and everything downstream is
built on sand. (Project Step 18.)

Local, CPU, no service. Needs paper_matrix.npy, paper_ids.npy, corpus parquet.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from sklearn.preprocessing import normalize


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
MATRIX_PATH = os.path.join(BASE, "paper_matrix.npy")
IDS_PATH    = os.path.join(BASE, "paper_ids.npy")
CORPUS      = os.path.join(BASE, "cleaned_corpus.parquet")
MIN_PER_CLASS = 500   # drop tiny fields that can't train/test meaningfully

def extract_primary_field(cell):
    if cell is None: return 'Unknown'
    try:
        items = list(cell)
        return items[0]['category'] if items else 'Unknown'
    except (TypeError, KeyError, IndexError):
        return 'Unknown'

# --- load embeddings + ids ---
X_all = np.load(MATRIX_PATH).astype('float32')
X_all = normalize(X_all)                      # unit-norm, same as retrieval space
paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
id_to_idx = {p: i for i, p in enumerate(paper_ids)}

# --- build labels from corpus ---
c = pd.read_parquet(CORPUS)
c['field'] = c['s2FieldsOfStudy'].apply(extract_primary_field)
pid_to_field = dict(zip(c['paperId'].astype(str), c['field']))

# align: only papers we have both a vector AND a field for
rows, labels = [], []
for pid, idx in id_to_idx.items():
    f = pid_to_field.get(pid)
    if f and f != 'Unknown':
        rows.append(idx); labels.append(f)

X = X_all[rows]
y = np.array(labels)
print(f"probing on {len(y)} papers, {len(set(y))} fields")

# --- drop fields too small to be meaningful ---
counts = pd.Series(y).value_counts()
keep = set(counts[counts >= MIN_PER_CLASS].index)
mask = np.array([lab in keep for lab in y])
X, y = X[mask], y[mask]
print(f"after dropping fields < {MIN_PER_CLASS}: {len(y)} papers, {len(set(y))} fields")

# --- majority-class baseline: what accuracy do you get by always guessing CS? ---
maj = pd.Series(y).value_counts(normalize=True).iloc[0]
print(f"majority-class baseline accuracy: {maj:.3f}")

# --- train/test split + logistic regression ---
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000, solver='saga')
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)

acc = accuracy_score(yte, pred)
bal_acc = balanced_accuracy_score(yte, pred)
print(f"\nlinear probe accuracy: {acc:.3f}")
print(f"linear probe balanced accuracy: {bal_acc:.3f}")
print(f"lift over majority baseline: {acc - maj:+.3f}")
print(f"lift over majority baseline (balanced): {bal_acc - maj:+.3f}")
print("\nper-field breakdown:")
print(classification_report(yte, pred, zero_division=0))