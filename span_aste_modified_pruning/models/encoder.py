import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Eq. 2
# bert-base-uncased
# google/bert_uncased_L-4_H-256_A-4
# google/bert_uncased_L-2_H-128_A-2

class SentenceEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', cache_dir = "span_aste/huggingface_models", use_bilstm=False, hidden_size=256):
        super(SentenceEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.bert = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.use_bilstm = use_bilstm

        if use_bilstm:
            self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)

    def forward(self, sentences, spans):
        tokenized = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        device = next(self.bert.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state.to(device)

        if self.use_bilstm:
            token_embeddings, _ = self.bilstm(token_embeddings)

        span_embeddings = []
        for i, sent_spans in enumerate(spans):
            sent_embeds = []
            for start, end in sent_spans:
                if end < start or end >= token_embeddings.shape[1]:
                    span_embed = torch.zeros(token_embeddings.shape[-1], device=device)
                else:
                    span_embed = token_embeddings[i, start:end+1].mean(dim=0)
                sent_embeds.append(span_embed)
            if sent_embeds:
                span_tensor = torch.stack(sent_embeds).to(device)
            else:
                span_tensor = torch.empty(0, device=device)
            span_embeddings.append(span_tensor)

        return span_embeddings
