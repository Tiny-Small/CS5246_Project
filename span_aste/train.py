import torch
from tqdm import tqdm
import pandas as pd
import os
import gc
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from models.encoder import SentenceEncoder
# from models.mention import MentionClassifier
# from models.triplet import TripletClassifier
# from data.dataset import setup_dataloader
# from utils import filter_valid_mentions, generate_candidate_pairs, compute_loss
# from evaluate import evaluate
# import config

from span_aste.models.encoder import SentenceEncoder
from span_aste.models.mention import MentionClassifier
from span_aste.models.triplet import TripletClassifier
from span_aste.data.dataset import setup_dataloader
from span_aste.utils import filter_valid_mentions, generate_candidate_pairs, compute_loss
from span_aste.evaluate import evaluate
import span_aste.config as config

# Limit candidate pairs to avoid memory explosion
MAX_TRIPLET_PAIRS = 5000

def train(df, val_df=None, epochs=3, batch_size=8, lr=2e-5, max_span_len=5, gamma=1.0, aspect_sentiment_only=False, save_dir="checkpoints", log_dir="logs", log_naming="train", load_checkpoint= None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SentenceEncoder().to(device)
    embedding_dim = encoder.bert.config.hidden_size

    if aspect_sentiment_only:
        mention_model = MentionClassifier(input_dim=embedding_dim, num_classes=2).to(device)
    else:
        mention_model = MentionClassifier(input_dim=embedding_dim, num_classes=3).to(device)

    triplet_model = TripletClassifier(input_dim=embedding_dim, num_classes=config.NUM_CLASSES, aspect_sentiment_only=aspect_sentiment_only).to(device) ##

    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint)
        encoder.load_state_dict(checkpoint['encoder'])
        mention_model.load_state_dict(checkpoint['mention_model'])

    train_loader, _ = setup_dataloader(df, tokenizer=encoder.tokenizer, batch_size=batch_size, max_span_len=max_span_len, aspect_sentiment_only=aspect_sentiment_only) ##
    val_loader = None
    if val_df is not None:
        val_loader, _ = setup_dataloader(val_df, tokenizer=encoder.tokenizer, batch_size=batch_size, max_span_len=max_span_len, aspect_sentiment_only=aspect_sentiment_only) ##

    optimizer = torch.optim.AdamW([
        {'params': encoder.bert.parameters(), 'lr': 1e-5},
        {'params': mention_model.parameters(), 'lr': lr},
        {'params': triplet_model.parameters(), 'lr': lr}
    ])

    # Learning rate scheduler
    encoder_scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5,
    )

    mention_scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )

    triplet_scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )


    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_1 = os.path.join(log_dir, f"{log_naming}.log")
    log_2 = os.path.join(log_dir, f"{log_naming}.csv")
    log_file = open(log_1, "w")
    epoch_logs = []

    for epoch in range(epochs):
        encoder.eval()
        mention_model.train()
        triplet_model.train()
        total_loss = 0
        total_mention_loss = 0
        total_triplet_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            print(f"[Epoch {epoch+1}] Processing batch {batch_idx+1}/{len(train_loader)}")

            sentences = batch['texts']
            spans = batch['spans']
            mention_targets = [torch.tensor(labels, dtype=torch.long, device=device) for labels in batch['mention_labels']]

            if aspect_sentiment_only:
                aspect_labels = batch['aspect_labels'] # this is a dictionary[aspect_span] = label
            else:
                triplet_labels = batch['triplet_labels']

            span_embeddings = encoder(sentences, spans)
            mention_logits = mention_model(span_embeddings)

            with torch.no_grad():
                mention_probs = [torch.softmax(logits, dim=-1) for logits in mention_logits]
            filtered_mentions = filter_valid_mentions(spans, mention_probs, span_embeddings)

            if aspect_sentiment_only:
                batch_aspects = []
                aspect_targets = []

                for i, aspect_dict in enumerate(aspect_labels):
                    # Grab only the aspect-labeled spans
                    for span, label in aspect_dict.items():
                        # Find its corresponding embedding
                        span_embed_list = span_embeddings[i]
                        span_list = spans[i]

                        try:
                            idx = span_list.index(span)
                        except ValueError:
                            continue

                        embed = span_embed_list[idx]
                        batch_aspects.append(embed)
                        aspect_targets.append(label)

                if batch_aspects:
                    aspect_logits = triplet_model(batch_aspects)
                    aspect_targets = torch.tensor(aspect_targets, dtype=torch.long, device=device)
                else:
                    aspect_logits = torch.empty(0, config.NUM_CLASSES).to(device)
                    aspect_targets = torch.empty(0, dtype=torch.long, device=device)

            else:

                batch_triplets = []
                triplet_targets = []
                for i, triplets in enumerate(triplet_labels):
                    label_map = {(t, o): label for (t, o, label) in triplets}
                    candidates = generate_candidate_pairs(filtered_mentions[i])

                    if not candidates:
                        span_list = spans[i]  # List of (start, end) spans
                        span_embed_list = span_embeddings[i]  # Tensor of span embeddings (num_spans, hidden_size)

                        max_negatives = min(20, len(span_list) ** 2)
                        negative_pairs = set()

                        while len(negative_pairs) < max_negatives:
                            t_idx = random.randint(0, len(span_list) - 1)
                            o_idx = random.randint(0, len(span_list) - 1)
                            if t_idx == o_idx:
                                continue
                            if (t_idx, o_idx) in negative_pairs:
                                continue

                            t_span = span_list[t_idx]
                            o_span = span_list[o_idx]
                            t_embed = span_embed_list[t_idx]
                            o_embed = span_embed_list[o_idx]

                            distance = min(abs(t_span[1] - o_span[0]), abs(t_span[0] - o_span[1]))

                            label = label_map.get((t_span, o_span), config.EMOTION_LABELS['INVALID'])
                            batch_triplets.append(((t_span, o_span), t_embed, o_embed, distance))
                            triplet_targets.append(label)
                            negative_pairs.add((t_idx, o_idx))

                    else:
                        for (t_span, o_span), t_embed, o_embed, distance in candidates:
                            label = label_map.get((t_span, o_span), config.EMOTION_LABELS['INVALID'])
                            batch_triplets.append(((t_span, o_span), t_embed, o_embed, distance))
                            triplet_targets.append(label)

                if len(batch_triplets) > MAX_TRIPLET_PAIRS:
                    print(f"Capped from {len(batch_triplets)} to {MAX_TRIPLET_PAIRS} triplets in batch {batch_idx+1}")
                    sampled_indices = random.sample(range(len(batch_triplets)), MAX_TRIPLET_PAIRS)
                    batch_triplets = [batch_triplets[i] for i in sampled_indices]
                    triplet_targets = [triplet_targets[i] for i in sampled_indices]

                if batch_triplets:
                    triplet_logits = triplet_model(batch_triplets)
                    triplet_targets = torch.tensor(triplet_targets, dtype=torch.long, device=device)
                else:
                    triplet_logits = torch.empty(0, config.NUM_CLASSES).to(device)
                    triplet_targets = torch.empty(0, dtype=torch.long, device=device)

            # Triplet / Aspect alignment
            if aspect_sentiment_only:
                emotions_logits = aspect_logits
                emotions_targets = aspect_targets
            else:
                emotions_logits = triplet_logits
                emotions_targets = triplet_targets

            loss, mention_loss, triplet_loss = compute_loss(
                mention_logits, mention_targets,
                emotions_logits, emotions_targets,
                use_focal=True,
                use_focal_for_triplet=True,
                aspect_sentiment_only=aspect_sentiment_only,
                gamma=gamma)

            print(f"[Epoch {epoch+1}] Batch {batch_idx+1} Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(mention_model.parameters()) + list(triplet_model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            total_mention_loss += mention_loss.item()
            total_triplet_loss += triplet_loss.item()

            # Memory monitoring
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / 1024**2
                reserved = torch.cuda.memory_reserved(device) / 1024**2
                print(f"[GPU MEM] Batch {batch_idx+1}: Allocated={allocated:.2f}MB | Reserved={reserved:.2f}MB")

            # Memory cleanup
            if aspect_sentiment_only:
                del span_embeddings, mention_logits, mention_probs, aspect_logits, aspect_targets, mention_targets
            else:
                del span_embeddings, mention_logits, mention_probs, batch_triplets, triplet_logits, triplet_targets, mention_targets
            torch.cuda.empty_cache()
            gc.collect()

            if aspect_sentiment_only:
                epoch_msg = f"Epoch {epoch+1} / {epochs} | Loss: {total_loss:.4f} | Mention Loss: {total_mention_loss:.4f} | Aspect Loss: {total_triplet_loss:.4f}"
            else:
                epoch_msg = f"Epoch {epoch+1} / {epochs} | Loss: {total_loss:.4f} | Mention Loss: {total_mention_loss:.4f} | Triplet Loss: {total_triplet_loss:.4f}"
            print(epoch_msg)

            if val_loader is not None:
                print("\nRunning validation...")
                with torch.no_grad():
                    output_metrics = evaluate(val_loader, encoder, mention_model, triplet_model, aspect_sentiment_only)

        mention_scheduler.step(output_metrics['macro_f1_mention_gold_only'])
        triplet_scheduler.step(output_metrics['macro_f1_triplet_gold_only'])
        encoder_scheduler.step(output_metrics['micro_triplet_gold_only']['f1'])
        current_encoder_lr = mention_scheduler.optimizer.param_groups[0]['lr']
        current_mention_lr = mention_scheduler.optimizer.param_groups[1]['lr']
        current_triplet_lr = mention_scheduler.optimizer.param_groups[2]['lr']
        print(f"[Scheduler] Current LR Encoder after Epoch {epoch+1}: {current_encoder_lr:.2e}")
        print(f"[Scheduler] Current LR Mention after Epoch {epoch+1}: {current_mention_lr:.2e}")
        print(f"[Scheduler] Current LR Triplet after Epoch {epoch+1}: {current_triplet_lr:.2e}")

        # Save per-epoch metrics
        epoch_logs.append({
            "epoch": epoch + 1,
            "loss": total_loss,
            "mention_loss": total_mention_loss,
            "triplet_or_aspect_loss": total_triplet_loss,
            "macro_f1_mention": output_metrics['macro_f1_mention_all'],
            "macro_f1_triplet": output_metrics['macro_f1_triplet_all'],
            "micro_precision": output_metrics['micro_triplet_all']['precision'],
            "micro_recall": output_metrics['micro_triplet_all']['recall'],
            "micro_f1": output_metrics['micro_triplet_all']['f1'],

            "macro_f1_mention_gold_only": output_metrics['macro_f1_mention_gold_only'],
            "macro_f1_triplet_gold_only": output_metrics['macro_f1_triplet_gold_only'],
            "micro_precision_gold_only": output_metrics['micro_triplet_gold_only']['precision'],
            "micro_recall_gold_only": output_metrics['micro_triplet_gold_only']['recall'],
            "micro_f1_gold_only": output_metrics['micro_triplet_gold_only']['f1'],

            "macro_f1_mention": output_metrics['macro_f1_mention_gold_pred'],
            "macro_f1_triplet": output_metrics['macro_f1_triplet_gold_pred'],
            "micro_precision_gold_pred": output_metrics['micro_triplet_gold_pred']['precision'],
            "micro_recall_gold_pred": output_metrics['micro_triplet_gold_pred']['recall'],
            "micro_f1_gold_pred": output_metrics['micro_triplet_gold_pred']['f1'],

            "lr_encoder": current_encoder_lr,
            "lr_mention": current_mention_lr,
            "lr_triplet": current_triplet_lr
        })

        log_file.write(epoch_msg + "\n")
        log_file.flush()

        # Save model
        torch.save({
            'encoder': encoder.state_dict(),
            'mention_model': mention_model.state_dict(),
            'triplet_model': triplet_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }, f"{save_dir}/checkpoint_epoch{epoch+1}.pt")

    log_file.close()

    # Save metrics to CSV
    metrics_df = pd.DataFrame(epoch_logs)
    metrics_df.to_csv(log_2, index=False)
    print(f"[INFO] Saved per-epoch metrics to {log_2}")
    print("\n=== Final Evaluation Metrics ===")
    print(metrics_df.tail(1).T)
