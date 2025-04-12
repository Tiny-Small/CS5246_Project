import torch
import torch.nn as nn
import torch.nn.functional as F

DISTANCE_BUCKETS = [0, 1, 2, 3, 4, 5, 7, 10, 14, 20, 30, 64]  # Similar to paper
DISTANCE_EMBEDDING_DIM = 128  # Same as paper

def bucket_distance(d):
    for i, b in enumerate(DISTANCE_BUCKETS):
        if d <= b:
            return i
    return len(DISTANCE_BUCKETS)

class TripletClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, num_classes=4):
        super(TripletClassifier, self).__init__()
        self.distance_embeddings = nn.Embedding(len(DISTANCE_BUCKETS) + 1, DISTANCE_EMBEDDING_DIM)

        in_dim = input_dim * 2 + DISTANCE_EMBEDDING_DIM

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.4)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        for layer in [self.linear1, self.linear2, self.linear3, self.output]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, target_opinion_pairs):
        features = []
        for (span_pair, t_embed, o_embed, raw_distance) in target_opinion_pairs:
            if torch.isnan(t_embed).any() or torch.isnan(o_embed).any():
                continue  # skip this pair

            bucketed = bucket_distance(raw_distance)
            dist_embed = self.distance_embeddings(torch.tensor(bucketed, device=t_embed.device))

            concat = torch.cat([t_embed, o_embed, dist_embed], dim=-1)
            if torch.isnan(concat).any():
                continue
            features.append(concat)

        if not features:
            return torch.empty(0, 8, device=next(self.parameters()).device)

        x = torch.stack(features).to(next(self.parameters()).device)

        x1 = self.dropout(self.activation(self.linear1(x)))
        x2 = self.dropout(self.activation(self.linear2(x1))) + x1
        x3 = self.dropout(self.activation(self.linear3(x2))) + x2
        logits = self.output(x3)

        return logits
