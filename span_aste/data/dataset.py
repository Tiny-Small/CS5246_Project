from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader

# from data.preprocess import preprocess_row, group_rows_by_text_df

from span_aste.data.preprocess import preprocess_row, group_rows_by_text_df

class ASTEPretrainingDataset(Dataset):
    def __init__(self, grouped_dataset, tokenizer, max_span_len=5, aspect_sentiment_only=False):
        self.data = grouped_dataset
        self.tokenizer = tokenizer
        self.max_span_len = max_span_len
        self.aspect_sentiment_only = aspect_sentiment_only

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return preprocess_row(
            row,
            tokenizer=self.tokenizer,
            max_span_len=self.max_span_len,
            filtered_spans=True,
            aspect_sentiment_only=self.aspect_sentiment_only
        )

def collate_fn(batch, aspect_sentiment_only):
    texts = [item['text'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    spans = [item['spans'] for item in batch]
    mention_labels = [item['mention_labels'] for item in batch]

    if aspect_sentiment_only:
        aspect_labels = [item['aspect_labels'] for item in batch]
        return {
            'texts': texts,
            'tokens': tokens,
            'spans': spans,
            'mention_labels': mention_labels,
            'aspect_labels': aspect_labels
        }
    else:
        triplet_labels = [item['triplet_labels'] for item in batch]
        return {
            'texts': texts,
            'tokens': tokens,
            'spans': spans,
            'mention_labels': mention_labels,
            'triplet_labels': triplet_labels
        }

# ---------- Dataloader Setup ----------

def setup_dataloader(df, tokenizer, batch_size=8, max_span_len=5, aspect_sentiment_only=False):
    grouped_dataset = group_rows_by_text_df(df, aspect_sentiment_only)
    dataset = ASTEPretrainingDataset(
        grouped_dataset=grouped_dataset,
        tokenizer=tokenizer,
        max_span_len=max_span_len,
        aspect_sentiment_only=aspect_sentiment_only
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, aspect_sentiment_only=aspect_sentiment_only)
    )
    return dataloader, tokenizer
