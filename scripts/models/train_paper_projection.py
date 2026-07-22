"""
Train ONLY a new paper projection nn.Linear(768 -> 256), Option 1 (controlled).

Central hypothesis under test: the two-tower model loses to a mean-pool baseline because the
paper tower's 768->256 projection was never trained (the live index is frozen SciBERT under a
RANDOM projection). Here we train that projection and nothing else, so the single changed
variable is the paper space.

GEOMETRY (approved -- Option A, decoupled):
  The frozen user tower was trained on the random-projection space R (== paper_matrix.npy ==
  embedding_cache.pt stacked). It consumes R vectors as history input and its output space is
  aligned to R. We therefore:
    * anchor   = frozen_user_tower(R history)      -- exactly the query used at eval, NO grad
    * positive = new_projection(paper_768[pos])    -- trainable
    * negative = new_projection(paper_768[neg])    -- trainable, in-batch negative
  i.e. we pull the trained paper space INTO the frozen user tower's fixed output space. At eval
  the user tower keeps reading R (query unchanged from baseline); only the search index becomes
  the trained space. This never feeds the frozen tower out-of-distribution inputs.

Objective is reproduced exactly from two_tower.py:
    F.triplet_margin_loss(anchor, positive, negative, margin=0.2, p=2, reduction='none').mean()
  No L2/normalization is added to the paper embeddings before the loss. (The anchor is unit-norm
  only because that normalization is intrinsic to the frozen user tower's forward -- it is the
  eval query, not an extra step.)

Reuses repo code: UserHistoryEncoder (scripts.eval.evaluate, the normalized/eval definition),
sample_negatives + collate_cached (scripts.models.two_tower). Torch only, never FAISS.
MPS for the model forward; in-batch negative SAMPLING is done on CPU (avoids the MPS
Generator limitation) and is deterministic at val for comparable early-stopping.

Writes scripts/models/paper_projection_trained.pt (state_dict of the trained Linear). Touches
no existing file.
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

from scripts.eval.evaluate import UserHistoryEncoder            # normalized (eval) tower # noqa: E402
from scripts.models.two_tower import sample_negatives, collate_cached  # noqa: E402

# ---- CONFIG (mirrors two_tower.py __main__) ----
BASE        = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
HISTORIES   = os.path.join(BASE, "histories.parquet")
R_MATRIX    = os.path.join(BASE, "paper_matrix.npy")          # random-proj 256-d (history input space)
P768        = os.path.join(BASE, "paper_768_full.npy")        # raw 768-d pooled SciBERT (positive source)
IDS_PATH    = os.path.join(BASE, "paper_ids.npy")             # row alignment for BOTH matrices
USER_CKPT   = os.path.join(REPO_ROOT, "scripts", "models", "user_history_encoder_best.pt")
OUT_PROJ    = os.path.join(REPO_ROOT, "scripts", "models", "paper_projection_trained.pt")

EMBED_DIM   = 256
IN_DIM      = 768
BATCH_SIZE  = 64
LR          = 1e-3
MARGIN      = 0.2
MAX_EPOCHS  = 8
PATIENCE    = 3
VAL_SEED    = 1234   # deterministic val negatives -> comparable val loss across epochs


class PaperProjTripletDataset(Dataset):
    """
    Like two_tower.CachedTripletDataset, but history and positive come from DIFFERENT spaces:
      history  -> R (normalized paper_matrix.npy rows)   [what the frozen tower expects]
      positive -> raw 768-d paper_768_full.npy row       [what the new projection maps]
    Returns the same item keys collate_cached expects. Negatives are sampled in-batch later.
    """
    def __init__(self, histories_df, id_to_idx, R_norm, P_raw):
        self.data = histories_df.reset_index(drop=True)
        self.id_to_idx = id_to_idx
        self.R = R_norm       # (N, 256) normalized
        self.P = P_raw        # (N, 768) raw
        self.chain_to_int = {c: i for i, c in enumerate(self.data['chain_id'].unique())}

    def __len__(self):
        return len(self.data)

    def _hist(self, pid):
        idx = self.id_to_idx.get(str(pid))
        return self.R[idx] if idx is not None else torch.zeros(EMBED_DIM)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        hist = torch.stack([self._hist(p) for p in row['history_ids']])   # (L, 256)
        pos_idx = self.id_to_idx[str(row['positive_id'])]                 # pre-filtered to exist
        return {
            'history_embeddings': hist,
            'positive_embedding': self.P[pos_idx],                        # (768,)
            'history_len': hist.shape[0],
            'chain_int': self.chain_to_int[row['chain_id']],
        }


def sample_negatives_cpu(positive, chain_ints, seed=None):
    """
    Run the repo's in-batch same-chain-guarded negative sampler on CPU (MPS has no torch
    Generator), returning negatives on the caller's device. Deterministic when seed is given.
    """
    dev = positive.device
    pos_cpu = positive.cpu()
    chain_cpu = chain_ints.cpu()
    gen = None
    if seed is not None:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
    neg_cpu, valid_cpu = sample_negatives(pos_cpu, chain_cpu, generator=gen)
    return neg_cpu.to(dev), valid_cpu.to(dev)


def run_epoch(loader, user_tower, paper_proj, device, optimizer=None, val_seed=None, log_every=2000):
    """One pass. Train when optimizer is given; else eval (deterministic negatives via val_seed)."""
    train = optimizer is not None
    paper_proj.train(train)
    total, nb = 0.0, 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        hist = batch['history_embeddings'].to(device)
        mask = batch['history_mask'].to(device)
        pos  = batch['positive_embedding'].to(device)     # (B, 768) raw
        chain = batch['chain_ints'].to(device)
        B = pos.shape[0]
        if B < 2:
            continue

        with torch.no_grad():                              # frozen tower -> eval query, no grad
            anchor, _ = user_tower(hist, mask)             # (B, 256), unit-norm

        if train:
            neg, valid = sample_negatives(pos, chain)      # on-device (random), same-chain guard
        else:
            neg, valid = sample_negatives_cpu(pos, chain, seed=val_seed)  # deterministic val
        if valid.sum() == 0:
            continue

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            pos_proj = paper_proj(pos)                     # (B, 256)
            neg_proj = paper_proj(neg)                     # (B, 256)
            losses = F.triplet_margin_loss(anchor, pos_proj, neg_proj,
                                           margin=MARGIN, p=2, reduction='none')
            loss = losses[valid].mean()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total += loss.item()
        nb += 1
        if train and log_every and nb % log_every == 0:
            rate = nb / (time.time() - t0)
            print(f"    [{'train' if train else 'val'}] batch {nb}  "
                  f"running_loss={total/nb:.4f}  {rate:.1f} batch/s", flush=True)
    return total / max(nb, 1)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # ---- load spaces (row-aligned to paper_ids.npy) ----
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    R_norm = F.normalize(torch.from_numpy(np.load(R_MATRIX).astype('float32')), p=2, dim=1)
    P_raw  = torch.from_numpy(np.load(P768).astype('float32'))
    assert R_norm.shape == (len(paper_ids), EMBED_DIM), R_norm.shape
    assert P_raw.shape  == (len(paper_ids), IN_DIM), P_raw.shape
    print(f"R (history input, normalized): {tuple(R_norm.shape)}   "
          f"P768 (positive source, raw): {tuple(P_raw.shape)}", flush=True)

    # ---- frozen user tower (normalized/eval definition) ----
    user_tower = UserHistoryEncoder(embedding_dim=EMBED_DIM).to(device)
    user_tower.load_state_dict(torch.load(USER_CKPT, map_location=device))
    user_tower.eval()
    for p in user_tower.parameters():
        p.requires_grad = False

    # ---- fresh trainable projection ----
    paper_proj = nn.Linear(IN_DIM, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(paper_proj.parameters(), lr=LR)

    # ---- PROVE the freeze: only the projection is optimized ----
    user_grad = sum(p.requires_grad for p in user_tower.parameters())
    proj_params = sum(p.numel() for p in paper_proj.parameters())
    opt_params  = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    print("\n--- freeze check ---")
    print(f"user-tower params with requires_grad=True : {user_grad}  (want 0)")
    print(f"paper_proj param count                     : {proj_params}  (768*256+256 = {IN_DIM*EMBED_DIM+EMBED_DIM})")
    print(f"optimizer param count                      : {opt_params}  (must == paper_proj)")
    assert user_grad == 0, "user tower is NOT frozen"
    assert opt_params == proj_params == IN_DIM*EMBED_DIM+EMBED_DIM, "optimizer holds more than the projection"
    print("--- only the paper projection is trainable ---\n", flush=True)

    # ---- data (same split the repo uses) ----
    hist_df = pd.read_parquet(HISTORIES)
    train_df = hist_df[hist_df['split'] == 'train'].reset_index(drop=True)
    val_df   = hist_df[hist_df['split'] == 'val'].reset_index(drop=True)

    def drop_missing_pos(df):
        keep = df['positive_id'].astype(str).isin(id_to_idx)
        return df[keep].reset_index(drop=True), int((~keep).sum())
    train_df, tdrop = drop_missing_pos(train_df)
    val_df,   vdrop = drop_missing_pos(val_df)
    print(f"train rows: {len(train_df)} (dropped {tdrop} w/ unknown positive)   "
          f"val rows: {len(val_df)} (dropped {vdrop})", flush=True)

    train_ds = PaperProjTripletDataset(train_df, id_to_idx, R_norm, P_raw)
    val_ds   = PaperProjTripletDataset(val_df,   id_to_idx, R_norm, P_raw)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_cached, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_cached, drop_last=True)

    # ---- train w/ early stopping on val loss ----
    best_val, no_improve, best_epoch = float('inf'), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, user_tower, paper_proj, device, optimizer=optimizer)
        va = run_epoch(val_loader, user_tower, paper_proj, device, optimizer=None, val_seed=VAL_SEED)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}  train={tr:.4f}  val={va:.4f}  ({dt:.1f}s)", flush=True)
        if va < best_val - 1e-6:
            best_val, no_improve, best_epoch = va, 0, epoch
            torch.save(paper_proj.state_dict(), OUT_PROJ)
            print(f"           new best (val={best_val:.4f}) -> saved {OUT_PROJ}", flush=True)
        else:
            no_improve += 1
            print(f"           no improvement ({no_improve}/{PATIENCE})", flush=True)
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (plateaued). Best epoch {best_epoch}, val={best_val:.4f}")
                break
    else:
        print(f"Reached MAX_EPOCHS={MAX_EPOCHS}. Best epoch {best_epoch}, val={best_val:.4f}")

    print(f"\nSaved trained projection -> {OUT_PROJ}  (best epoch {best_epoch}, val={best_val:.4f})")


if __name__ == "__main__":
    main()
