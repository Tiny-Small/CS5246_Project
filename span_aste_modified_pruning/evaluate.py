import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from span_aste.utils import EMOTION_LABELS, filter_valid_mentions, generate_candidate_pairs, compute_micro_f1

def evaluate(dataloader, encoder, mention_model, triplet_model):
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

            # Triplet classification metrics
            for i, triplets in enumerate(triplet_labels):
                #print(f"[Eval Batch {i}] {len(filtered_mentions[i])} spans after filtering")

                num_targets = sum(1 for s in filtered_mentions[i] if s[1] == 0)
                num_opinions = sum(1 for s in filtered_mentions[i] if s[1] == 1)
                #print(f"  --> Targets: {num_targets}, Opinions: {num_opinions}")

                label_map = {(t, o): label for (t, o, label) in triplets if label != EMOTION_LABELS['INVALID']}
                candidates = generate_candidate_pairs(filtered_mentions[i])
                #print(f"  --> Candidate pairs: {len(candidates)}")

                if not candidates:
                    #print(f"[Eval Batch {i}] No valid target-opinion pairs generated — skipping triplet evaluation.")
                    all_gold_triplets.append([(t, o, l) for (t, o, l) in triplets if l != EMOTION_LABELS['INVALID']])
                    all_pred_triplets.append([])
                    continue

                gold_triplets = [(t, o, l) for (t, o, l) in triplets if l != EMOTION_LABELS['INVALID']]
                pred_triplets = []

                for (t_span, o_span), t_embed, o_embed, distance in candidates:
                    gold_label = label_map.get((t_span, o_span), EMOTION_LABELS['INVALID'])
                    logits = triplet_model([((t_span, o_span), t_embed, o_embed, distance)])
                    pred_label = logits.argmax(dim=-1).item()
                    # print(pred_label)
                    all_triplet_targets.append(gold_label)
                    all_triplet_preds.append(pred_label)

                    if pred_label != EMOTION_LABELS['INVALID']:
                        pred_triplets.append((t_span, o_span, pred_label))

                all_gold_triplets.append(gold_triplets)
                all_pred_triplets.append(pred_triplets)

    print(f"[DEBUG] Total gold aspect spans: {sum(t == 0 for t in all_mention_targets)}")
    print(f"[DEBUG] Total gold opinion spans: {sum(t == 1 for t in all_mention_targets)}")
    print(f"[DEBUG] Total predicted aspect spans: {sum(p == 0 for p in all_mention_preds)}")
    print(f"[DEBUG] Total predicted opinion spans: {sum(p == 1 for p in all_mention_preds)}")

    # Mention metrics
    print("\nMention Classification Report:")
    macro_f1_mention = 0.0
    if all_mention_targets and all_mention_preds:
        print(classification_report(all_mention_targets, all_mention_preds, digits=4, zero_division=0))
        macro_f1_mention = f1_score(all_mention_targets, all_mention_preds, average="macro", zero_division=0)
    else:
        print("[Mention Classification Report] Skipped — no predictions or targets.")
        print(f"[DEBUG] Total triplets predicted: {len(all_mention_preds)}, targets: {len(all_mention_targets)}")

    # Triplet metrics
    print("\nTriplet Classification Report:")
    macro_f1_triplet = 0.0
    if all_triplet_targets and all_triplet_preds:
        print(classification_report(all_triplet_targets, all_triplet_preds, digits=4, zero_division=0))
        macro_f1_triplet = f1_score(all_triplet_targets, all_triplet_preds, average="macro", zero_division=0)
    else:
        print("[Triplet Classification Report] Skipped — no predictions or targets.")
        print(f"[DEBUG] Total triplets predicted: {len(all_triplet_preds)}, targets: {len(all_triplet_targets)}")

    print("\nMicro F1 on Triplet Extraction:")
    p, r, f1 = compute_micro_f1(all_pred_triplets, all_gold_triplets)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

    return {
    "macro_f1_mention": macro_f1_mention,
    "macro_f1_triplet": macro_f1_triplet,
    "micro_f1_triplet": {"precision": p, "recall": r, "f1": f1}
    }
