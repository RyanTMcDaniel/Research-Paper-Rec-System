"""
CAPSTONE: co-train BOTH towers from scratch on the trained paper space.

Diagnosis from prior experiments: the frozen user tower was entangled with the random-projection
geometry, so improving the paper space alone broke query-index alignment. Fix implied: train the
user tower AND a fresh paper projection JOINTLY, so their geometries co-adapt into ONE space.

  * User tower : fresh UserHistoryEncoder (evaluate.py's normalized architecture == 'current').
  * Paper tower: fresh nn.Linear(768 -> 256) applied to precomputed paper_768_full.npy.
                 SciBERT is NOT instantiated here -> frozen by construction (only the projection
                 is the trainable paper side).
  * Both trainable modules in ONE optimizer -> 525,824 trainable params, trained together.

SINGLE-GEOMETRY (this is the whole point, vs the prior decoupled experiment):
  history input to the tower = normalize(proj(paper_768[hist]))   -- the SAME transform eval uses
  anchor   = user_tower(that)                                     -- unit-norm fingerprint
  positive = proj(paper_768[pos])   (raw)                         -- no pre-loss normalization
  negative = proj(paper_768[neg])   (raw), chosen in-batch semi-hard
Query and index therefore live in one co-trained geometry by construction; at eval the index is
proj(all papers) and the query is user_tower(normalize(proj(history))) -- consistent.

Objective reproduced exactly from two_tower.py:
  F.triplet_margin_loss(anchor, positive, negative, margin=0.2, p=2, reduction='none').mean()

NEGATIVES: in-batch SEMI-HARD -- a FAISS-free approximation of full-corpus hard mining. Within a
batch of B=512, for each anchor pick the CLOSEST other-chain in-batch paper that is still farther
than the true positive (semi-hard); fall back to hardest-valid, then drop if no valid candidate.
Repo has no torch hard-neg path and FAISS is forbidden, so this is the correct approximation.
(Random in-batch -- the repo default -- is the trivially-easy flaw this project diagnosed.)

Torch only, never FAISS. MPS for the forward/backward; tiny in-batch mining done on CPU.
Saves scripts/models/user_tower_cotrained.pt and scripts/models/paper_projection_cotrained.pt.
Touches no existing file.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from scripts.eval.evaluate import UserHistoryEncoder      # normalized (eval) architecture # noqa: E402
from scripts.models.two_tower import collate_cached        # per-batch padding + mask # noqa: E402

BASE       = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES  = os.path.join(BASE, "histories.parquet")
P768       = os.path.join(BASE, "paper_768_full.npy")
IDS_PATH   = os.path.join(BASE, "paper_ids.npy")
OUT_USER   = os.path.join(REPO_ROOT, "scripts", "models", "user_tower_cotrained.pt")
OUT_PROJ   = os.path.join(REPO_ROOT, "scripts", "models", "paper_projection_cotrained.pt")

EMBED_DIM, IN_DIM = 256, 768
BATCH_SIZE = 512
LR         = 1e-3
MARGIN     = 0.2
MAX_EPOCHS = 8
PATIENCE   = 3
EPOCH1_BUDGET_S = 25 * 60   # auto-pause guard: if epoch 1 exceeds this, stop and report


class CoTrainDataset(Dataset):
    """history + positive are BOTH raw 768-d rows (projected inside the training loop)."""
    def __init__(self, df, id_to_idx, P_raw):
        self.data = df.reset_index(drop=True)
        self.id_to_idx = id_to_idx
        self.P = P_raw
        self.chain_to_int = {c: i for i, c in enumerate(self.data['chain_id'].unique())}

    def __len__(self):
        return len(self.data)

    def _row(self, pid):
        idx = self.id_to_idx.get(str(pid))
        return self.P[idx] if idx is not None else torch.zeros(IN_DIM)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        hist = torch.stack([self._row(p) for p in row['history_ids']])   # (L, 768)
        return {
            'history_embeddings': hist,
            'positive_embedding': self.P[self.id_to_idx[str(row['positive_id'])]],  # (768,)
            'history_len': hist.shape[0],
            'chain_int': self.chain_to_int[row['chain_id']],
        }


def semihard_negatives(anchor, pos_proj, chain_ints):
    """
    In-batch semi-hard mining (FAISS-free approximation of full-corpus hard mining).
    anchor, pos_proj: (B, 256) on device. Returns (neg_proj, valid_mask) on device.
    Selection uses DETACHED distances (index picking only); gradients flow via the gather.
    """
    a = anchor.detach().cpu()
    p = pos_proj.detach().cpu()
    chain = chain_ints.cpu()
    B = a.shape[0]

    D = torch.cdist(a, p)                       # (B,B) anchor_i -> positive_j
    d_pos = D.diagonal().unsqueeze(1)           # (B,1)
    invalid = (chain[:, None] == chain[None, :]) | torch.eye(B, dtype=torch.bool)

    # semi-hard: valid AND farther than the positive -> closest such
    Dsh = D.clone()
    Dsh[invalid] = float('inf')
    Dsh[D <= d_pos] = float('inf')
    has_sh = torch.isfinite(Dsh).any(1)

    # fallback: hardest valid (closest overall among other-chain)
    Dh = D.clone()
    Dh[invalid] = float('inf')
    has_valid = torch.isfinite(Dh).any(1)

    chosen = torch.where(has_sh, Dsh.argmin(1), Dh.argmin(1))   # (B,)
    chosen = torch.where(has_valid, chosen, torch.zeros_like(chosen))  # safe index; masked out
    neg = pos_proj[chosen.to(pos_proj.device)]                  # gather WITH grad
    return neg, has_valid.to(anchor.device)


def run_epoch(loader, user_tower, proj, device, optimizer=None, log_every=500):
    train = optimizer is not None
    user_tower.train(train); proj.train(train)
    total, nb = 0.0, 0
    t0 = time.time()
    ctx_outer = torch.enable_grad() if train else torch.no_grad()
    with ctx_outer:
        for batch in loader:
            hist = batch['history_embeddings'].to(device)     # (B, L, 768)
            mask = batch['history_mask'].to(device)           # (B, L)
            pos  = batch['positive_embedding'].to(device)     # (B, 768)
            chain = batch['chain_ints'].to(device)
            B = pos.shape[0]
            if B < 2:
                continue

            # SINGLE geometry: history input = normalize(proj(hist_768)), same as eval
            hist_input = F.normalize(proj(hist), p=2, dim=-1)  # (B, L, 256)
            anchor, _ = user_tower(hist_input, mask)           # (B, 256) unit-norm
            pos_proj = proj(pos)                               # (B, 256) raw

            neg_proj, valid = semihard_negatives(anchor, pos_proj, chain)
            if valid.sum() == 0:
                continue

            losses = F.triplet_margin_loss(anchor, pos_proj, neg_proj,
                                           margin=MARGIN, p=2, reduction='none')
            loss = losses[valid].mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item(); nb += 1
            if train and log_every and nb % log_every == 0:
                print(f"    [train] batch {nb}  running_loss={total/nb:.4f}  "
                      f"{nb/(time.time()-t0):.1f} batch/s", flush=True)
    return total / max(nb, 1)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    P_raw = torch.from_numpy(np.load(P768).astype('float32'))
    assert P_raw.shape == (len(paper_ids), IN_DIM), P_raw.shape

    user_tower = UserHistoryEncoder(embedding_dim=EMBED_DIM).to(device)   # fresh init
    proj = nn.Linear(IN_DIM, EMBED_DIM).to(device)                        # fresh init
    params = list(user_tower.parameters()) + list(proj.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    n_user = sum(p.numel() for p in user_tower.parameters())
    n_proj = sum(p.numel() for p in proj.parameters())
    n_opt  = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    print("\n--- trainable params ---")
    print(f"user tower : {n_user}")
    print(f"projection : {n_proj}")
    print(f"optimizer  : {n_opt}  (want {n_user + n_proj} = 525824; BERT not instantiated -> frozen)")
    assert n_opt == n_user + n_proj == 525824, "optimizer param count mismatch"
    print("negatives  : in-batch SEMI-HARD (FAISS-free approximation of full-corpus hard mining)\n", flush=True)

    hist_df = pd.read_parquet(HISTORIES)
    train_df = hist_df[hist_df['split'] == 'train'].reset_index(drop=True)
    val_df   = hist_df[hist_df['split'] == 'val'].reset_index(drop=True)
    for name, df in (("train", train_df), ("val", val_df)):
        assert df['positive_id'].astype(str).isin(id_to_idx).all(), f"{name} has unknown positive"
    print(f"train triplets: {len(train_df)}   val triplets: {len(val_df)}   batch: {BATCH_SIZE}", flush=True)

    train_loader = DataLoader(CoTrainDataset(train_df, id_to_idx, P_raw), batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_cached, drop_last=True)
    val_loader   = DataLoader(CoTrainDataset(val_df, id_to_idx, P_raw), batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_cached, drop_last=True)  # deterministic

    best_val, no_improve, best_epoch = float('inf'), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, user_tower, proj, device, optimizer=optimizer)
        va = run_epoch(val_loader, user_tower, proj, device, optimizer=None)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}  train={tr:.4f}  val={va:.4f}  ({dt:.1f}s = {dt/60:.1f} min)", flush=True)

        if va < best_val - 1e-6:
            best_val, no_improve, best_epoch = va, 0, epoch
            torch.save(user_tower.state_dict(), OUT_USER)
            torch.save(proj.state_dict(), OUT_PROJ)
            print(f"           new best (val={best_val:.4f}) -> saved cotrained user tower + projection", flush=True)
        else:
            no_improve += 1
            print(f"           no improvement ({no_improve}/{PATIENCE})", flush=True)
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best epoch {best_epoch}, val={best_val:.4f}")
                break

        if epoch == 1 and dt > EPOCH1_BUDGET_S:
            print(f"\n*** PAUSE: epoch-1 wall-clock {dt/60:.1f} min exceeds the {EPOCH1_BUDGET_S/60:.0f} min "
                  f"budget. Stopping after 1 epoch as instructed -- report before grinding through 8. ***")
            print(f"(best-so-far checkpoint saved at epoch {best_epoch}, val={best_val:.4f})")
            return
    else:
        print(f"Reached MAX_EPOCHS={MAX_EPOCHS}. Best epoch {best_epoch}, val={best_val:.4f}")

    print(f"\nSaved cotrained checkpoints (best epoch {best_epoch}, val={best_val:.4f}):")
    print(f"  {OUT_USER}\n  {OUT_PROJ}")


if __name__ == "__main__":
    main()
