import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

# import config
# from utils import filter_valid_mentions, generate_candidate_pairs, compute_micro, get_mention_labels

import span_aste.config as config
from span_aste.utils import filter_valid_mentions, generate_candidate_pairs, compute_micro, get_mention_labels


DUMMY_INVALID = -999

def evaluate(dataloader, encoder, mention_model, triplet_model, aspect_sentiment_only=False):
    encoder.eval()
    mention_model.eval()
    triplet_model.eval()

    all_mention_targets, all_mention_preds = [], []
    all_triplet_targets, all_triplet_preds = [], []
    all_gold_triplets, all_pred_triplets = [], []

    with torch.no_grad():
        for batch in dataloader:
            sentences = batch['texts']
            spans = batch['spans']
            mention_labels = batch['mention_labels']

            if aspect_sentiment_only:
                aspect_labels = batch['aspect_labels']
            else:
                triplet_labels = batch['triplet_labels']

            mention_targets = [labels for labels in mention_labels]
            span_embeddings = encoder(sentences, spans)
            mention_logits = mention_model(span_embeddings)

            # Mention classification metrics
            for pred_logits, target in zip(mention_logits, mention_targets):
                if pred_logits.numel() == 0:
                    continue
                preds = pred_logits.argmax(dim=-1)
                all_mention_targets.extend(target)
                all_mention_preds.extend(preds.cpu())

            mention_probs = [torch.softmax(logits, dim=-1) for logits in mention_logits]
            filtered_mentions = filter_valid_mentions(spans, mention_probs, span_embeddings)

            if aspect_sentiment_only:
                for i, aspect_dict in enumerate(aspect_labels):
                    aspects = [s for s, label in aspect_dict.items() if label != config.EMOTION_LABELS['INVALID']]
                    labels = [label for s, label in aspect_dict.items() if label != config.EMOTION_LABELS['INVALID']]
                    gold_aspects = list(zip(aspects, labels))
                    pred_aspects = []

                    for span, label in aspect_dict.items():
                        for s, pred_class, _ in filtered_mentions[i]:
                            if s == span and pred_class == 0:
                                embed = span_embeddings[i][spans[i].index(span)]
                                logits = triplet_model([embed])
                                pred_label = logits.argmax(dim=-1).item()
                                pred_aspects.append((span, pred_label))
                                all_triplet_targets.append(label)
                                all_triplet_preds.append(pred_label)
                                break

                    all_gold_triplets.append(gold_aspects)
                    all_pred_triplets.append(pred_aspects)


            else:
                # Triplet classification metrics
                for i, triplets in enumerate(triplet_labels):
                    print(f"[Eval Batch {i}] {len(filtered_mentions[i])} spans after filtering")

                    num_targets = sum(1 for s in filtered_mentions[i] if s[1] == 0)
                    num_opinions = sum(1 for s in filtered_mentions[i] if s[1] == 1)
                    print(f"  --> Targets: {num_targets}, Opinions: {num_opinions}")

                    label_map = {(t, o): label for (t, o, label) in triplets if label != config.EMOTION_LABELS['INVALID']}
                    candidates = generate_candidate_pairs(filtered_mentions[i])
                    print(f"  --> Candidate pairs: {len(candidates)}")

                    if not candidates:
                        print(f"[Eval Batch {i}] No valid target-opinion pairs generated — skipping triplet evaluation.")
                        all_gold_triplets.append([(t, o, l) for (t, o, l) in triplets if l != config.EMOTION_LABELS['INVALID']])
                        all_pred_triplets.append([])
                        continue

                    gold_triplets = [(t, o, l) for (t, o, l) in triplets if l != config.EMOTION_LABELS['INVALID']]
                    pred_triplets = []

                    for (t_span, o_span), t_embed, o_embed, distance in candidates:
                        gold_label = label_map.get((t_span, o_span), config.EMOTION_LABELS['INVALID'])
                        logits = triplet_model([((t_span, o_span), t_embed, o_embed, distance)])
                        pred_label = logits.argmax(dim=-1).item()
                        all_triplet_targets.append(gold_label)
                        all_triplet_preds.append(pred_label)

                        if pred_label != config.EMOTION_LABELS['INVALID']:
                            pred_triplets.append((t_span, o_span, pred_label))

                    all_gold_triplets.append(gold_triplets)
                    all_pred_triplets.append(pred_triplets)

    print(f"[DEBUG] Total gold aspect spans: {sum(t == 0 for t in all_mention_targets)}")
    print(f"[DEBUG] Total gold opinion spans: {sum(t == 1 for t in all_mention_targets)}")
    print(f"[DEBUG] Total predicted aspect spans: {sum(p == 0 for p in all_mention_preds)}")
    print(f"[DEBUG] Total predicted opinion spans: {sum(p == 1 for p in all_mention_preds)}")

    mention_label_map = get_mention_labels(aspect_sentiment_only)

    # No Drop Invalids
    print("\n==No Drop Invalids==")
    macro_f1_mention = evaluate_classification("Mention", all_mention_targets, all_mention_preds,
                                               invalid_label=DUMMY_INVALID, drop_invalid=False, both=False)
    macro_f1_triplet = evaluate_classification("Triplet", all_triplet_targets, all_triplet_preds,
                                               invalid_label=DUMMY_INVALID, drop_invalid=False, both=False)
    micro_result = evaluate_micro_f1_triplets(all_pred_triplets, all_gold_triplets,
                                              invalid_label=DUMMY_INVALID, drop_invalid=False, both=False)

    # Drop Invalids from Gold Target
    print("\n==Drop Invalids from Gold Target==")
    macro_f1_mention_D1 = evaluate_classification("Mention", all_mention_targets, all_mention_preds,
                                                  invalid_label=mention_label_map["INVALID"], drop_invalid=True, both=False)
    macro_f1_triplet_D1 = evaluate_classification("Triplet", all_triplet_targets, all_triplet_preds,
                                                  invalid_label=config.EMOTION_LABELS['INVALID'], drop_invalid=True, both=False)
    micro_result_D1 = evaluate_micro_f1_triplets(all_pred_triplets, all_gold_triplets,
                                                 invalid_label=config.EMOTION_LABELS['INVALID'], drop_invalid=True, both=False)

    # Drop Invalids from Gold Target and Pred
    print("\n==Drop Invalids from Gold Target and Pred==")
    macro_f1_mention_D2 = evaluate_classification("Mention", all_mention_targets, all_mention_preds,
                                                  invalid_label=mention_label_map["INVALID"], drop_invalid=True, both=True)
    macro_f1_triplet_D2 = evaluate_classification("Triplet", all_triplet_targets, all_triplet_preds,
                                                  invalid_label=config.EMOTION_LABELS['INVALID'], drop_invalid=True, both=True)
    micro_result_D2 = evaluate_micro_f1_triplets(all_pred_triplets, all_gold_triplets,
                                                 invalid_label=config.EMOTION_LABELS['INVALID'], drop_invalid=True, both=True)

    return {
        "macro_f1_mention_all": macro_f1_mention,
        "macro_f1_triplet_all": macro_f1_triplet,
        "micro_triplet_all": micro_result,

        "macro_f1_mention_gold_only": macro_f1_mention_D1,
        "macro_f1_triplet_gold_only": macro_f1_triplet_D1,
        "micro_triplet_gold_only": micro_result_D1,

        "macro_f1_mention_gold_pred": macro_f1_mention_D2,
        "macro_f1_triplet_gold_pred": macro_f1_triplet_D2,
        "micro_triplet_gold_pred": micro_result_D2,
    }


def drop_invalids(all_targets, all_preds, invalid_label: int, both: bool = False):
    """
    Filters out entries with the specified invalid class from the target
    and optionally the predictions.

    Args:
        all_targets (List[int]): List of ground truth labels.
        all_preds (List[int]): List of predicted labels.
        invalid_class (int): Class index representing 'INVALID'.
        both (bool): If True, remove entries where either target or pred == invalid_class.
                     If False, remove entries where only target == invalid_class.

    Returns:
        Tuple[List[int], List[int]]: Filtered targets and predictions.
    """
    if both:
        valid_pairs = [
            (gold, pred) for gold, pred in zip(all_targets, all_preds)
            if gold != invalid_label and pred != invalid_label
        ]
    else:
        valid_pairs = [
            (gold, pred) for gold, pred in zip(all_targets, all_preds)
            if gold != invalid_label
        ]

    targets = [g for g, _ in valid_pairs]
    preds   = [p for _, p in valid_pairs]

    return targets, preds



def evaluate_classification(name, golds, preds, invalid_label : int, drop_invalid: bool = False, both: bool = False):
    """
    Evaluate classification metrics with optional INVALID label filtering.

    Args:
        name (str): Label name for printing, e.g., "Mention", "Triplet"
        golds (List[int]): Gold labels
        preds (List[int]): Predicted labels
        drop_invalid (bool): Whether to drop 'INVALID' from evaluation
        both (bool): If True, drop 'INVALID' from both golds and preds. If False, only drop from golds.

    Returns:
        macro_f1 (float): Computed macro-F1 score
    """
    print(f"\n{name} Classification Report:")

    if drop_invalid:
        golds, preds = drop_invalids(golds, preds, invalid_label=invalid_label , both=both)

    if not golds or not preds:
        print(f"[{name} Classification Report] Skipped — no predictions or targets.")
        print(f"[DEBUG] Total {name.lower()} predicted: {len(preds)}, targets: {len(golds)}")
        return 0.0

    print(classification_report(golds, preds, digits=4, zero_division=0))
    macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)
    return macro_f1

def evaluate_micro_f1_triplets(pred_triplets, gold_triplets, invalid_label : int, drop_invalid: bool = False, both: bool = False):
    print("\nMicro F1 on Triplet Extraction:")

    if drop_invalid:
        gold_triplets, pred_triplets = drop_invalids(gold_triplets, pred_triplets, invalid_label=invalid_label, both=both)

    p, r, f1 = compute_micro(pred_triplets, gold_triplets)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    return {"precision": p, "recall": r, "f1": f1}
