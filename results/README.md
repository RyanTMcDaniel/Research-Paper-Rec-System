# results

`training_log_hardneg.csv` is the per-epoch log of the run that produced the
shipped checkpoint, `scripts/models/user_history_encoder_best.pt` (normalized
geometry + chain-guarded hard negatives, 4 epochs, early stopping fired after
epoch 1's best validation loss). Every number in README.md and ABLATION.md
traces to that run.

`legacy/` holds the logs of two earlier random-negative runs (8 epochs each,
val loss ~0.065 scale). They are superseded, produced no shipped artifact, and
back no number in the docs — kept only as the loss-scale comparison referenced
in ABLATION.md's provenance note.
