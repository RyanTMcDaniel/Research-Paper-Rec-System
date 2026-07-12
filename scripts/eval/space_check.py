import os

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.environ.get("RPR_DATA_DIR", os.path.join(REPO_ROOT, "data", "training_data"))
CKPT = os.path.join(REPO_ROOT, "scripts", "models", "user_history_encoder_best.pt")

class UserHistoryEncoder(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, h, m):
        kpm = ~m.bool()
        a, _ = self.attention(h, h, h, key_padding_mask=kpm)
        mask = m.unsqueeze(-1).float()
        pooled = (a * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.projection(pooled), _

matrix = np.load(os.path.join(BASE, "paper_matrix.npy")).astype("float32")
paper_ids = [str(p) for p in np.load(os.path.join(BASE, "paper_ids.npy"), allow_pickle=True)]
id_to_idx = {p: i for i, p in enumerate(paper_ids)}
matrix_t = torch.from_numpy(matrix)

model = UserHistoryEncoder()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

# --- scale of raw paper vectors ---
paper_norms = np.linalg.norm(matrix, axis=1)
print(f"raw paper vec norm:   mean={paper_norms.mean():.3f}  std={paper_norms.std():.3f}")

# --- build a few real fingerprints and check their scale + neighbors ---
h = pd.read_parquet(os.path.join(BASE, "histories.parquet"))
test = h[h['split']=='test'].sample(5, random_state=1)

for _, row in test.iterrows():
    hist = [str(x) for x in row['history_ids'] if str(x) in id_to_idx]
    if not hist: continue
    hv = matrix_t[[id_to_idx[x] for x in hist]]
    L = hv.shape[0]

    with torch.no_grad():
        fp, _ = model(hv.unsqueeze(0), torch.ones(1, L))
    fp = fp.squeeze(0)
    mp = hv.mean(0)

    print(f"\nhistory len {L}")
    print(f"  model fingerprint norm: {fp.norm():.3f}")
    print(f"  meanpool   vector norm: {mp.norm():.3f}")
    # distance from each to the nearest raw paper
    d_fp = torch.cdist(fp.unsqueeze(0), matrix_t).min().item()
    d_mp = torch.cdist(mp.unsqueeze(0), matrix_t).min().item()
    print(f"  nearest-paper dist  model={d_fp:.3f}   meanpool={d_mp:.3f}")