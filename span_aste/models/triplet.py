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
        # num_classes=4 represents the sentiment relation between Aspect/Target and Opinion: {Positive, Negative, Neutral, Invalid}
        # Invalid indicates that the target-opinion pair has no valid sentiment relationship.
        # num_classes=7 represents 'NEUTRAL', 'SADNESS', 'JOY', 'ANGER', 'DISGUST', 'FEAR', 'SURPRISE', 'INVALID'
        super(TripletClassifier, self).__init__()
        self.distance_embeddings = nn.Embedding(len(DISTANCE_BUCKETS) + 1, DISTANCE_EMBEDDING_DIM)

        # self.ffn = nn.Sequential(
        #     nn.Linear(input_dim * 2 + DISTANCE_EMBEDDING_DIM, hidden_dim),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Dropout(0.4),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Dropout(0.4),
        #     nn.Linear(hidden_dim, num_classes)
        # )
        # # Xavier initialization
        # for layer in self.ffn:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)
        #         nn.init.constant_(layer.bias, 0.0)

        in_dim = input_dim * 2 + DISTANCE_EMBEDDING_DIM

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.4)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # Xavier initialization
        for layer in [self.linear1, self.linear2, self.linear3, self.output]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)


    def forward(self, target_opinion_pairs):
        features = []
        # for (_, t_embed, o_embed, distance) in target_opinion_pairs:
        #     distance = float(distance)
        #     if not torch.isfinite(torch.tensor(distance)):
        #         distance = 0.0
        #     distance_tensor = torch.tensor([distance], dtype=t_embed.dtype, device=t_embed.device)
        for (span_pair, t_embed, o_embed, raw_distance) in target_opinion_pairs:
            if torch.isnan(t_embed).any() or torch.isnan(o_embed).any():
                continue  # skip this pair

            bucketed = bucket_distance(raw_distance)
            dist_embed = self.distance_embeddings(torch.tensor(bucketed, device=t_embed.device))

            # concat = torch.cat([t_embed, o_embed, distance_tensor], dim=-1)
            concat = torch.cat([t_embed, o_embed, dist_embed], dim=-1)
            if torch.isnan(concat).any():
                continue
            features.append(concat)

        # Skip forward if too many pairs to avoid OOM
        # if len(features) > 10000:
        #     print(f"[TripletClassifier] Skipping batch with {len(features)} candidate pairs to avoid OOM.")
        #     return torch.empty(0, 8, device=next(self.parameters()).device)

        if not features:
            # device = next(self.parameters()).device
            return torch.empty(0, 8, device=next(self.parameters()).device)

        # feature_tensor = torch.stack(features).to(next(self.parameters()).device)
        # logits = self.ffn(feature_tensor)

        x = torch.stack(features).to(next(self.parameters()).device)

        # Residual forward
        x1 = self.dropout(self.activation(self.linear1(x)))
        x2 = self.dropout(self.activation(self.linear2(x1))) + x1
        x3 = self.dropout(self.activation(self.linear3(x2))) + x2
        logits = self.output(x3)
        return logits
