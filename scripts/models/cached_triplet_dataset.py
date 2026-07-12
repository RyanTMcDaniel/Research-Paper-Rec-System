import torch
from torch.utils.data import Dataset

class CachedTripletDataset(Dataset):
    """
    Reads precomputed embeddings from embedding_cache instead of tokenizing.
    No max_history_len truncation — histories keep their natural length so the
    overlap/overfit structure stays intact for the popularity-bias audit.
    Padding happens per-batch in the collate_fn, not here.

    embedding_cache: dict {paperId: tensor(256,)} from embedding_cache.pt
    No negative_id column — negatives sampled in-batch in the training loop.
    """
    def __init__(self, histories_df, embedding_cache, embedding_dim=256):
        self.data = histories_df.reset_index(drop=True)
        self.cache = embedding_cache
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.data)

    def _lookup(self, pid):
        emb = self.cache.get(pid)
        if emb is None:
            return torch.zeros(self.embedding_dim)
        return emb

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # full natural-length history, no truncation
        history_embs = [self._lookup(pid) for pid in row['history_ids']]
        history_tensor = torch.stack(history_embs)          # (L_i, D) — variable L_i

        pos_emb = self._lookup(row['positive_id'])          # (D,)

        return {
            'history_embeddings': history_tensor,           # (L_i, D)
            'positive_embedding': pos_emb,                  # (D,)
            'history_len': history_tensor.shape[0],
        }