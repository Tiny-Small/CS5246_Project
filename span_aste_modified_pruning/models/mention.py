import torch
import torch.nn as nn

# Eq. 3
class MentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, num_classes=3):
        super(MentionClassifier, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )

        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, span_embeddings_batch):
        device = next(self.parameters()).device
        all_logits = []
        for span_embeds in span_embeddings_batch:
            if span_embeds.numel() == 0:
                all_logits.append(torch.empty(0, 3, device=device))
            else:
                logits = self.ffn(span_embeds.to(device))
                all_logits.append(logits)
        return all_logits
