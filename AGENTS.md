# Project: Research Paper Recommender

Two-tower recommender (frozen SciBERT paper tower + self-attention user tower)
over a 253K-paper corpus. The headline result is negative: mean-pool beats the
learned tower, and ABLATION.md documents why. Python + PyTorch.

## Layout
- `scripts/data_collection/` — corpus/citation fetching and cleaning pipeline
- `scripts/models/` — `two_tower.py` (model defs + Kaggle training entrypoint)
- `scripts/retrieval/` — embedding-cache export and FAISS index build
- `scripts/eval/` — benchmark and the six ablation experiments
- `scripts/tests/` — standalone check scripts (run directly with python; not pytest)

## Constraints
- `data/` and `*.pt` are gitignored artifacts. Never commit them; never
  regenerate or overwrite them without being asked — eval results depend on
  the exact shipped versions.
- Never import torch and faiss in the same process (OpenMP crash on Apple
  Silicon). Eval scripts search in plain torch for this reason.
- API keys come from the `S2_API_KEY` env var only. Never hardcode.

## Style
- Prefer explicit over clever. Type hints on new functions.
- Name generated artifacts after the config that produced them
  (e.g. `training_log_hardneg.csv`).
