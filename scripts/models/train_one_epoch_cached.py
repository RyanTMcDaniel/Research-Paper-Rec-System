from scripts.models.train import triplet_loss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def collate_cached(batch):
    """
    Collate function for CachedTripletDataset.
    Pads histories to the max length in the batch and creates a mask.
    """
    max__len = max(item['history_len'] for item in batch)
    D = batch[0]['history_embeddings'].shape[1]

    history_batch = []
    mask_batch = []
    positive_batch = []

    for item in batch:
        h = item['history_embeddings']
        L = h.shape[0]
        pad = max__len - L
        if pad > 0:
            h = torch.cat([h, torch.zeros(pad, D)], dim=0)
        history_batch.append(h)

        mask = torch.zeros(max__len)
        mask[:L] = 1.0
        mask_batch.append(mask)

        positive_batch.append(item['positive_embedding'])

    return {
        'history_embeddings': torch.stack(history_batch),  # (B, L_max, D`
        'history_mask': torch.stack(mask_batch),          # (B, L_max)
        'positive_embedding': torch.stack(positive_batch),  # (B, D)
    }


def train_one_epoch_cached(history_encoder, dataloader, optimizer, device, margin=0.2):
    """
    Cached training loop. No tokenizer, no PaperEncoder forward pass.
    Paper embeddings are precomputed (frozen SciBERT) and read from the batch.
    Negatives sampled in-batch: each anchor's negative is another sample's positive.
    """
    history_encoder.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        history_embeddings = batch['history_embeddings'].to(device)  # (B, L_i, D)
        history_mask = batch['history_mask'].to(device)                # (B, L_i)
        positive = batch['positive_embedding'].to(device)  # (B, D)

        anchor, _ = history_encoder(history_embeddings, history_mask)  # (B, D)

        B = positive.shape[0]
        if B < 2:
            continue  # skip batches with less than 2 samples

        neg_idx = (torch.arange(B, device=device) + 1) % B
        negative = positive[neg_idx]  # (B, D)

        loss = F.triplet_margin_loss(anchor, positive, negative, margin=margin, reduction='mean')
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)