import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# ---------- MODEL DEFINITIONS ----------

class PaperEncoder(nn.Module):
    """Kept for re-embedding at FAISS time (Step 9). Not used in cached training."""
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", output_dim=256):
        super().__init__()
        from transformers import AutoModel  # imported locally: only needed at FAISS re-embed time
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        counts = torch.clamp(counts, min=1e-9)
        mean_pooled = summed / counts

        embedding = self.projection(mean_pooled)
        return embedding


class UserHistoryEncoder(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, history_embeddings, history_mask):
        key_padding_mask = ~history_mask.bool()

        attn_output, attn_weights = self.attention(
            query=history_embeddings,
            key=history_embeddings,
            value=history_embeddings,
            key_padding_mask=key_padding_mask
        )

        mask = history_mask.unsqueeze(-1).float()
        summed = (attn_output * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        counts = torch.clamp(counts, min=1e-9)
        mean_pooled = summed / counts

        user_embedding = self.projection(mean_pooled)
        return user_embedding, attn_weights


# ---------- CACHED DATASET ----------

class CachedTripletDataset(Dataset):
    """
    Reads precomputed paper embeddings from embedding_cache instead of tokenizing.
    No max_history_len truncation -- natural history length preserved so the
    overlap/overfit structure stays intact for the popularity-bias audit.
    Padding happens per-batch in collate_cached.

    Also returns chain_id so the training loop can avoid same-chain false
    negatives during in-batch negative sampling.

    embedding_cache: dict {paperId: tensor(256,)}
    No negative_id -- negatives sampled in-batch in the training loop.
    """
    def __init__(self, histories_df, embedding_cache, embedding_dim=256):
        self.data = histories_df.reset_index(drop=True)
        self.cache = embedding_cache
        self.embedding_dim = embedding_dim

        # map each distinct chain_id to an integer so it can ride along as a tensor
        self.chain_to_int = {c: i for i, c in enumerate(self.data['chain_id'].unique())}

    def __len__(self):
        return len(self.data)

    def _lookup(self, pid):
        emb = self.cache.get(pid)
        if emb is None:
            return torch.zeros(self.embedding_dim)
        return emb

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        history_embs = [self._lookup(pid) for pid in row['history_ids']]
        history_tensor = torch.stack(history_embs)          # (L_i, D)

        pos_emb = self._lookup(row['positive_id'])          # (D,)

        return {
            'history_embeddings': history_tensor,
            'positive_embedding': pos_emb,
            'history_len': history_tensor.shape[0],
            'chain_int': self.chain_to_int[row['chain_id']],
        }


def collate_cached(batch):
    """Pad each batch to its own longest history; build the 1/0 mask."""
    max_len = max(item['history_len'] for item in batch)
    D = batch[0]['history_embeddings'].shape[1]

    history_batch, mask_batch, positive_batch, chain_batch = [], [], [], []

    for item in batch:
        h = item['history_embeddings']
        L = h.shape[0]
        pad = max_len - L
        if pad > 0:
            h = torch.cat([h, torch.zeros(pad, D)], dim=0)
        history_batch.append(h)

        mask = torch.zeros(max_len)
        mask[:L] = 1.0
        mask_batch.append(mask)

        positive_batch.append(item['positive_embedding'])
        chain_batch.append(item['chain_int'])

    return {
        'history_embeddings': torch.stack(history_batch),   # (B, max_len, D)
        'history_mask': torch.stack(mask_batch),            # (B, max_len)
        'positive_embedding': torch.stack(positive_batch),  # (B, D)
        'chain_ints': torch.tensor(chain_batch),            # (B,)
    }


# ---------- NEGATIVE SAMPLING ----------

def sample_negatives(positive, chain_ints, generator=None):
    """
    In-batch negatives via random permutation, with a same-chain guard.

    For each anchor i, pick some other row j != i whose chain differs from i's,
    and use positive[j] as the negative. This avoids the systematic false
    negatives you'd get from torch.roll when overlapping sliding-window rows
    from the same citation chain land in the same batch.

    Returns (negative, valid_mask):
      negative    -- (B, D) chosen negative embeddings
      valid_mask  -- (B,) bool; False where no different-chain row existed in
                     the batch (rare). Those rows are dropped from the loss.
    """
    B = positive.shape[0]
    device = positive.device

    # random candidate for every row
    perm = torch.randperm(B, generator=generator, device=device)

    # a candidate is bad if it points to itself OR shares the anchor's chain
    same_self = perm == torch.arange(B, device=device)
    same_chain = chain_ints[perm] == chain_ints

    bad = same_self | same_chain

    # for bad picks, try to repair with a second random pass
    if bad.any():
        perm2 = torch.randperm(B, generator=generator, device=device)
        repair_ok = (perm2 != torch.arange(B, device=device)) & (chain_ints[perm2] != chain_ints)
        use_repair = bad & repair_ok
        perm = torch.where(use_repair, perm2, perm)
        bad = (perm == torch.arange(B, device=device)) | (chain_ints[perm] == chain_ints)

    # any still-bad rows get dropped from the loss this step
    valid_mask = ~bad
    negative = positive[perm]
    return negative, valid_mask


# ---------- TRAINING / EVAL ----------

def train_one_epoch_cached(history_encoder, dataloader, optimizer, device, margin=0.2):
    history_encoder.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        history_embeddings = batch['history_embeddings'].to(device)
        history_mask       = batch['history_mask'].to(device)
        positive           = batch['positive_embedding'].to(device)
        chain_ints         = batch['chain_ints'].to(device)

        B = positive.shape[0]
        if B < 2:
            continue

        anchor, _ = history_encoder(history_embeddings, history_mask)
        negative, valid_mask = sample_negatives(positive, chain_ints)

        if valid_mask.sum() == 0:
            continue  # whole batch was same-chain; skip (extremely rare)

        # per-sample triplet loss, then average only over valid rows
        losses = F.triplet_margin_loss(
            anchor, positive, negative, margin=margin, reduction='none'
        )
        loss = losses[valid_mask].mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_cached(history_encoder, dataloader, device, margin=0.2):
    history_encoder.eval()
    total_loss = 0.0
    n_batches = 0

    # deterministic negatives at eval so val loss is comparable across epochs
    gen = torch.Generator(device=device)

    with torch.no_grad():
        for batch in dataloader:
            history_embeddings = batch['history_embeddings'].to(device)
            history_mask       = batch['history_mask'].to(device)
            positive           = batch['positive_embedding'].to(device)
            chain_ints         = batch['chain_ints'].to(device)

            B = positive.shape[0]
            if B < 2:
                continue

            anchor, _ = history_encoder(history_embeddings, history_mask)

            gen.manual_seed(1234)  # same negatives every eval pass
            negative, valid_mask = sample_negatives(positive, chain_ints, generator=gen)
            if valid_mask.sum() == 0:
                continue

            losses = F.triplet_margin_loss(
                anchor, positive, negative, margin=margin, reduction='none'
            )
            total_loss += losses[valid_mask].mean().item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------- MAIN ----------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # ---- CONFIG ----
    BASE = '/kaggle/input/datasets/ryanthomasmcdaniel/research-paper-rec-proj'
    CACHE_PATH = '/kaggle/input/datasets/ryanthomasmcdaniel/embedding-cache/embedding_cache.pt'
    OUT_DIR = '/kaggle/working'
    BATCH_SIZE = 64
    LR = 1e-3
    MARGIN = 0.2
    MAX_EPOCHS = 8
    PATIENCE = 3

    # Flip to True for a fast 1,000-row plumbing check; False for the real run.
    SMOKE_TEST = False

    # ---- LOAD ----
    histories_df = pd.read_parquet(f'{BASE}/histories.parquet')
    cache = torch.load(CACHE_PATH)

    print('cache entries:', len(cache))
    print('empty histories:', int((histories_df['history_len'] == 0).sum()))  # want 0

    train_df = histories_df[histories_df['split'] == 'train'].reset_index(drop=True)
    val_df   = histories_df[histories_df['split'] == 'val'].reset_index(drop=True)

    if SMOKE_TEST:
        train_df = train_df.sample(n=1000, random_state=42).reset_index(drop=True)
        val_df   = val_df.sample(n=1000, random_state=42).reset_index(drop=True)

    print(f'train rows: {len(train_df)}   val rows: {len(val_df)}')

    # ---- DATA ----
    train_dataset = CachedTripletDataset(train_df, cache)
    val_dataset   = CachedTripletDataset(val_df, cache)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_cached, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_cached, drop_last=True)

    # ---- MODEL ----
    history_encoder = UserHistoryEncoder().to(device)
    optimizer = torch.optim.Adam(history_encoder.parameters(), lr=LR)

    # ---- TRAIN LOOP w/ early stopping + checkpointing ----
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(OUT_DIR, 'user_history_encoder_best.pt')

    history_log = []  # for the writeup: per-epoch losses

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch_cached(history_encoder, train_loader, optimizer, device, margin=MARGIN)
        val_loss   = evaluate_cached(history_encoder, val_loader, device, margin=MARGIN)

        elapsed = time.time() - epoch_start
        history_log.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'seconds': elapsed})
        print(f"epoch {epoch:2d}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)")

        torch.save(history_encoder.state_dict(), os.path.join(OUT_DIR, 'user_history_encoder_last.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(history_encoder.state_dict(), best_ckpt_path)
            print(f"           new best (val={best_val_loss:.4f}) -> saved checkpoint")
        else:
            epochs_without_improvement += 1
            print(f"           no improvement ({epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    pd.DataFrame(history_log).to_csv(os.path.join(OUT_DIR, 'training_log_hardneg.csv'), index=False)
    print(f"Per-epoch log saved to {os.path.join(OUT_DIR, 'training_log_hardneg.csv')}")