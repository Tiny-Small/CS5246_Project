import torch
from torch.utils.data import Dataset, DataLoader
from span_aste.data.preprocess import preprocess_row, group_rows_by_text_df

class ASTEPretrainingDataset(Dataset):
    def __init__(self, grouped_data, tokenizer, max_span_len=5):
        self.data = grouped_data
        self.tokenizer = tokenizer
        self.max_span_len = max_span_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return preprocess_row(row, self.tokenizer, self.max_span_len)

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    spans = [item['spans'] for item in batch]
    mention_labels = [item['mention_labels'] for item in batch]
    triplet_labels = [item['triplet_labels'] for item in batch]
    return {
        'texts': texts,
        'tokens': tokens,
        'spans': spans,
        'mention_labels': mention_labels,
        'triplet_labels': triplet_labels
    }

# ---------- Dataloader Setup ----------

def setup_dataloader(df, tokenizer, batch_size=8, max_span_len=5):
    grouped_dataset = group_rows_by_text_df(df)
    dataset = ASTEPretrainingDataset(grouped_dataset, tokenizer, max_span_len=max_span_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader, tokenizer
