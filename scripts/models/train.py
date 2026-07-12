import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModel, AutoTokenizer


# ---------- MODEL DEFINITIONS ----------

class PaperEncoder(nn.Module):
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", output_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)

        # freeze SciBERT initially - only train projection layer first
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
    
class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=2, model_name="allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.paper_encoder = PaperEncoder(model_name=model_name, output_dim=embedding_dim)
        self.history_encoder = UserHistoryEncoder(embedding_dim=embedding_dim, num_heads=num_heads)

    def encode_paper(self, input_ids, attention_mask):
        return self.paper_encoder(input_ids, attention_mask)
    
    def encode_history(self, history_embeddings, history_mask):
        user_embedding, attn_weights = self.history_encoder(history_embeddings, history_mask)
        return user_embedding, attn_weights
    
# ---------- TRIPLET LOSS ----------

def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    anchor: user-history embedding (batch_size, embedding_dim)
    positive: embedding of the paper the user actually read (batch_size, embedding_dim)
    negative: embedding of a paper the user did not read (batch_size, embedding_dim)
    """
    dist_pos = F.pairwise_distance(anchor, positive, p=2)
    dist_neg = F.pairwise_distance(anchor, negative, p=2)
    losses = F.relu(dist_pos - dist_neg + margin)
    return losses.mean()

# ---------- DATASET ----------

class TripletDataset(Dataset):
    """
    histories_df columns actually present:
    chain_id, chain_type, chain_year, history_ids, positive_id,
    history_len, positive_year, positive_primary_field,
    history_primary_fields, split

    No negative_id — negatives are sampled in-batch during training.
    """
    def __init__(self, histories_df, paper_lookup, tokenizer, max_history_len=5, max_text_len=512):
        self.data = histories_df.reset_index(drop=True)
        self.paper_lookup = paper_lookup
        self.tokenizer = tokenizer
        self.max_history_len = max_history_len
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)
    
    def _get_text(self, paper_id):
        paper = self.paper_lookup.get(paper_id, {"title":"", "abstract":""})
        return f"{paper['title']}. {paper['abstract']}"
    
    def _tokenize(self, text):
        encoded = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_text_len, return_tensors='pt'
        )

        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        history_ids = list(row['history_ids'])[:self.max_history_len]
        pad_len = self.max_history_len - len(history_ids)

        history_input_ids = []
        history_attention_masks = []
        for pid in history_ids:
            ids, mask = self._tokenize(self._get_text(pid))
            history_input_ids.append(ids)
            history_attention_masks.append(mask)

        for _ in range(pad_len):
            history_input_ids.append(torch.zeros(self.max_text_len, dtype=torch.long))
            history_attention_masks.append(torch.zeros(self.max_text_len, dtype=torch.long))

        history_mask = torch.tensor([1] * len(history_ids) + [0] * pad_len, dtype=torch.float)

        pos_ids, pos_mask = self._tokenize(self._get_text(row['positive_id']))

        return {
            "history_input_ids": torch.stack(history_input_ids),
            "history_attention_masks": torch.stack(history_attention_masks),
            "history_mask": history_mask,
            "positive_input_ids": pos_ids,
            "positive_attention_mask": pos_mask,
        }

# ---------- TRAINING LOOP ----------

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        batch_size, hist_len, text_len = batch['history_input_ids'].shape

        flat_input_ids = batch['history_input_ids'].view(-1, text_len).to(device)
        flat_attention_masks = batch['history_attention_masks'].view(-1, text_len).to(device)

        flat_embeddings = model.encode_paper(flat_input_ids, flat_attention_masks)
        history_embeddings = flat_embeddings.view(batch_size, hist_len, -1)

        history_mask = batch['history_mask'].to(device)
        anchor, _ = model.encode_history(history_embeddings, history_mask)

        positive = model.encode_paper(
            batch['positive_input_ids'].to(device),
            batch['positive_attention_mask'].to(device)
        )

        # in-batch negative sampling: shift positives by one position within the batch
        # so each anchor's "negative" is some OTHER example's positive paper
        negative = torch.roll(positive, shifts=1, dims=0)

        loss = triplet_loss(anchor, positive, negative)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------- EVALUATION ----------


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch_size, hist_len, text_len = batch['history_input_ids'].shape

            flat_input_ids = batch['history_input_ids'].view(-1, text_len).to(device)
            flat_attention_masks = batch['history_attention_masks'].view(-1, text_len).to(device)

            flat_embeddings = model.encode_paper(flat_input_ids, flat_attention_masks)
            history_embeddings = flat_embeddings.view(batch_size, hist_len, -1)

            history_mask = batch['history_mask'].to(device)
            anchor, _ = model.encode_history(history_embeddings, history_mask)

            positive = model.encode_paper(
                batch['positive_input_ids'].to(device),
                batch['positive_attention_mask'].to(device)
            )

            negative = torch.roll(positive, shifts=1, dims=0)

            loss = triplet_loss(anchor, positive, negative)
            total_loss += loss.item()

        return total_loss / len(dataloader)
# ---------- MAIN ----------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)

    histories_df = pd.read_parquet('data/training_data/histories.parquet')
    corpus_df = pd.read_parquet('data/cleaned_corpus.parquet')
    paper_lookup = corpus_df.set_index('paperId')[["title", 'abstract']].to_dict('index')

    train_df = histories_df[histories_df['split'] == 'train'].reset_index(drop=True)
    # train_df = train_df.sample(n=20, random_state=42).reset_index(drop=True)   # TEMP: small test slice

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    train_dataset = TripletDataset(train_df, paper_lookup, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = TwoTowerModel().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    val_df = histories_df[histories_df['split'] == 'val'].reset_index(drop=True)
    # val_df = val_df.sample(n=10, random_state=42).reset_index(drop=True)   # TEMP: small test slice

    val_dataset = TripletDataset(val_df, paper_lookup, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    best_val_loss = float('inf')
    patience = 3
    epochs_without_improvement = 0
    max_epochs = 20

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{max_epochs} — train loss: {train_loss:.4f} — val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'two_tower_model_best.pt')
            print("  --> new best model saved")
        else:
            epochs_without_improvement += 1
            print(f"  -> no improvement ({epochs_without_improvement}/{patience})")

        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print("Training complete. Best val loss: ", best_val_loss)
        