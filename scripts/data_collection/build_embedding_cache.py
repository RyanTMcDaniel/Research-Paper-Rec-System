import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from scripts.models.two_tower import PaperEncoder  # adjust import path if needed on Kaggle


def build_embedding_cache(corpus_path, paper_encoder, tokenizer, device, batch_size=64):
    """
    Precompute embeddings for every unique paper in the corpus once.
    Returns {paperId: embedding_tensor} where embedding_tensor is shape (256,)
    """
    df = pd.read_parquet(corpus_path)
    paper_encoder.eval()

    # Build title+abstract text, handle missing values defensively
    df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

    cache = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[start:start + batch_size]
            paper_ids = batch_df['paperId'].tolist()
            texts = batch_df['text'].tolist()

            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            embeddings = paper_encoder(
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask']
            )

            for pid, emb in zip(paper_ids, embeddings):
                cache[pid] = emb.cpu()

    return cache


if __name__ == "__main__":
    import time

    # ---- CONFIG ----
    FULL_CORPUS_PATH = "/kaggle/input/your-dataset-name/cleaned_corpus.parquet"  # fix this path on Kaggle
    MODEL_NAME = "allenai/scibert_scivocab_uncased"
    BATCH_SIZE = 64

    # Flip this to False once the test slice run looks clean
    TEST_MODE = True
    TEST_SLICE_SIZE = 200

    if TEST_MODE:
        CORPUS_PATH = "corpus_test_slice.parquet"
        OUTPUT_PATH = "embedding_cache_test.pt"
        print(f"TEST_MODE on — building a {TEST_SLICE_SIZE}-row slice from the full corpus")
        full_df = pd.read_parquet(FULL_CORPUS_PATH)
        full_df.head(TEST_SLICE_SIZE).to_parquet(CORPUS_PATH)
    else:
        CORPUS_PATH = FULL_CORPUS_PATH
        OUTPUT_PATH = "embedding_cache.pt"
        print("TEST_MODE off — running on the full corpus")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    paper_encoder = PaperEncoder(model_name=MODEL_NAME, output_dim=256).to(device)

    # Freeze backbone — caching is only valid while SciBERT is frozen
    for param in paper_encoder.bert.parameters():
        param.requires_grad = False

    start_time = time.time()
    cache = build_embedding_cache(CORPUS_PATH, paper_encoder, tokenizer, device, batch_size=BATCH_SIZE)
    elapsed = time.time() - start_time

    n_rows = len(pd.read_parquet(CORPUS_PATH))
    print(f"Cache built: {len(cache)} entries (expected {n_rows})")
    print(f"Elapsed: {elapsed:.1f}s for {n_rows} rows ({elapsed / n_rows:.3f}s/row)")

    if TEST_MODE:
        est_full_seconds = (elapsed / n_rows) * 253703
        print(f"Estimated full-corpus (253,703 rows) time: {est_full_seconds / 60:.1f} minutes")

    # Sanity check on a sample entry before trusting the whole thing
    sample_key = next(iter(cache))
    sample_val = cache[sample_key]
    print(f"Sample entry shape: {sample_val.shape}, dtype: {sample_val.dtype}, has NaN: {torch.isnan(sample_val).any().item()}")

    torch.save(cache, OUTPUT_PATH)
    print(f"Saved cache to {OUTPUT_PATH} — download this before the Kaggle session ends.")