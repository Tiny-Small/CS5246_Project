import torch
import torch.nn as nn
import torch.nn.functional as F

#EMOTION_LABELS = {
#    'NEUTRAL': 0,
#    'SADNESS': 1,
#    'JOY': 2,
#    'ANGER': 3,
#    'DISGUST': 4,
#    'FEAR': 5,
#    'SURPRISE': 6,
#   'INVALID': 7
#}

#EMOTION_LABELS = {
#    "SADNESS": 0,
#    "JOY": 1,
#    "ANGER": 2,
#    "DISGUST": 3,
#    "FEAR": 4,
#    "SURPRISE": 5,
#    "INVALID": 6
#}


EMOTION_LABELS = {
    "POSITIVE": 0,
    "NEGATIVE": 1,
    "NEUTRAL": 2,
    "INVALID": 3
}

# Filter span candidates based on mention predictions
def filter_valid_mentions(spans, mention_probs, span_embeddings):
    filtered = []
    for span_list, prob_list, embed_list in zip(spans, mention_probs, span_embeddings):
        candidates = []
        for idx, (span, probs, embed) in enumerate(zip(span_list, prob_list, embed_list)):
            pred_class = torch.argmax(probs).item()
            if pred_class in [0, 1]:  # 0=Target, 1=Opinion
                candidates.append((span, pred_class, embed))
        filtered.append(candidates)
    return filtered

# Generate all target-opinion candidate pairs with distances
def generate_candidate_pairs(mentions):
    target_opinion_pairs = []
    targets = [x for x in mentions if x[1] == 0]
    opinions = [x for x in mentions if x[1] == 1]
    for t_span, _, t_embed in targets:
        for o_span, _, o_embed in opinions:
            distance = min(abs(t_span[1] - o_span[0]), abs(t_span[0] - o_span[1]))
            target_opinion_pairs.append(((t_span, o_span), t_embed, o_embed, distance))
    return target_opinion_pairs

# ---------------------------
# Focal Loss for Mention Module
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of class weights, e.g., [0.5, 0.5, 0.1]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets = targets.view(-1)

        log_p = log_probs[range(len(targets)), targets]
        p = probs[range(len(targets)), targets]

        if self.alpha is not None:
            alpha_factor = self.alpha[targets]
        else:
            alpha_factor = 1.0

        loss = -alpha_factor * (1 - p) ** self.gamma * log_p

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ---------------------------
# Compute loss with optional Focal Loss
# ---------------------------
def compute_loss(
    mention_logits,
    mention_targets,
    triplet_logits,
    triplet_targets,
    use_focal=True,
    use_focal_for_triplet=True
):
    loss = 0
    mention_loss = 0
    triplet_loss = 0

    # Mention loss setup
    if use_focal:
        alpha_mention = torch.tensor([0.5, 0.5, 0.05], device=mention_logits[0].device)
        mention_loss_fn = FocalLoss(alpha=alpha_mention, gamma=1.0)
    else:
        weights = torch.tensor([0.5, 0.5, 0.05], device=mention_logits[0].device)
        mention_loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Triplet loss setup
    if use_focal_for_triplet:
        alpha_triplet = torch.tensor(
            #[1.0] * 6 + [0.1],  # Classes 0–6 = valid emotions, 7 = INVALID
            # [1.0] * 8,
            [1.0, 1.0, 1.0, 0.1],  # POS, NEG, NEU, INVALID
            device=triplet_targets.device if isinstance(triplet_targets, torch.Tensor) else mention_logits[0].device
        )
        triplet_loss_fn = FocalLoss(alpha=alpha_triplet, gamma=1.0)
    else:
        weights = torch.tensor([1.0, 1.0, 1.0, 0.05], device=mention_logits[0].device)
        #weights = torch.tensor([1.0] * 6 + [0.1], device=mention_logits[0].device)
        triplet_loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Mention loss (per example)
    for logits, targets in zip(mention_logits, mention_targets):
        if logits.numel() > 0:
            mention_loss += mention_loss_fn(logits, targets)

    # Triplet loss (batch-level)
    if triplet_logits.numel() > 0:
        triplet_loss += triplet_loss_fn(triplet_logits, triplet_targets)

    loss = mention_loss + triplet_loss
    return loss, mention_loss, triplet_loss

def compute_micro_f1(pred_triplets_batch, gold_triplets_batch):
    assert len(pred_triplets_batch) == len(gold_triplets_batch)

    correct = 0
    total_pred = 0
    total_gold = 0

    for preds, golds in zip(pred_triplets_batch, gold_triplets_batch):
        preds = set(preds)
        golds = set(golds)

        correct += len(preds & golds)
        total_pred += len(preds)
        total_gold += len(golds)

    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_gold if total_gold > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1
