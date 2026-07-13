import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "models"))

import torch
from two_tower import PaperEncoder, UserHistoryEncoder
from transformers import AutoTokenizer

print("=== Testing PaperEncoder ===")

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

sample_text = """
    A Two-Tower Approach to Research Paper Recommendation. 
    This paper presents a method for recommending academic 
    papers using dual encoder towers.
"""

inputs = tokenizer(sample_text, return_tensors='pt', 
            padding=True, truncation=True, max_length=512)

encoder = PaperEncoder()

with torch.no_grad():
    output = encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

print("Paper embedding shape: ", output.shape)


print("\n=== Testing UserHistoryEncoder ===")

history_encoder = UserHistoryEncoder()

fake_history = torch.randn(1, 4, 256)

fake_mask = torch.ones(1, 4)

with torch.no_grad():
    user_embedding, attn_weights = history_encoder(fake_history, fake_mask)

print("User embedding shape: ", user_embedding.shape)
print("Attention weights shape: ", attn_weights.shape)