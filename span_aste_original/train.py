import torch
from tqdm import tqdm
import pandas as pd
import os
import gc
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

from span_aste.models.encoder import SentenceEncoder
from span_aste.models.mention import MentionClassifier
from span_aste.models.triplet import TripletClassifier
from span_aste.data.dataset import setup_dataloader
from span_aste.utils import filter_valid_mentions, generate_candidate_pairs, compute_loss, EMOTION_LABELS
from span_aste.evaluate import evaluate

# Limit candidate pairs to avoid memory explosion
MAX_TRIPLET_PAIRS = 5000

def train(df, val_df=None, epochs=3, batch_size=8, lr=2e-5, max_span_len=5, save_dir="span_aste/checkpoints", log_path="span_aste/logs/train.log"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SentenceEncoder().to(device)
    embedding_dim = encoder.bert.config.hidden_size  # e.g., 768 or 128

    mention_model = MentionClassifier(input_dim=embedding_dim).to(device)
    triplet_model = TripletClassifier(input_dim=embedding_dim, num_classes=4).to(device)

    checkpoint = torch.load("/path/CS5246/span_aste/checkpoints/16_check/checkpoint_epoch100.pt")
    encoder.load_state_dict(checkpoint['encoder'])
    mention_model.load_state_dict(checkpoint['mention_model'])

    train_loader, _ = setup_dataloader(df, tokenizer=encoder.tokenizer, batch_size=batch_size, max_span_len=max_span_len)
    val_loader = None
    if val_df is not None:
        val_loader, _ = setup_dataloader(val_df, tokenizer=encoder.tokenizer, batch_size=batch_size, max_span_len=max_span_len)

    optimizer = torch.optim.AdamW([
        {'params': encoder.bert.parameters(), 'lr': 1e-5},  # Slower LR for BERT
        {'params': mention_model.parameters(), 'lr': lr},   # LR from args or default
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
    log_file = open(log_path, "w")

    for epoch in range(epochs):
        encoder.eval()
        mention_model.train()
        triplet_model.train()
        total_loss = 0
        total_mention_loss = 0
        total_triplet_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):

            sentences = batch['texts']
            spans = batch['spans']
            mention_targets = [torch.tensor(labels, dtype=torch.long, device=device) for labels in batch['mention_labels']]
            triplet_labels = batch['triplet_labels']

            span_embeddings = encoder(sentences, spans)
            mention_logits = mention_model(span_embeddings)

            with torch.no_grad():
                mention_probs = [torch.softmax(logits, dim=-1) for logits in mention_logits]
            filtered_mentions = filter_valid_mentions(spans, mention_probs, span_embeddings)

            batch_triplets = []
            triplet_targets = []
            for i, triplets in enumerate(triplet_labels):
                label_map = {(t, o): label for (t, o, label) in triplets}
                candidates = generate_candidate_pairs(filtered_mentions[i])

                if not candidates:
                    span_list = spans[i]
                    span_embed_list = span_embeddings[i]

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

                        label = label_map.get((t_span, o_span), EMOTION_LABELS['INVALID'])
                        batch_triplets.append(((t_span, o_span), t_embed, o_embed, distance))
                        triplet_targets.append(label)
                        negative_pairs.add((t_idx, o_idx))

                else:
                    for (t_span, o_span), t_embed, o_embed, distance in candidates:
                        label = label_map.get((t_span, o_span), EMOTION_LABELS['INVALID'])
                        batch_triplets.append(((t_span, o_span), t_embed, o_embed, distance))
                        triplet_targets.append(label)

            if len(batch_triplets) > MAX_TRIPLET_PAIRS:
                sampled_indices = random.sample(range(len(batch_triplets)), MAX_TRIPLET_PAIRS)
                batch_triplets = [batch_triplets[i] for i in sampled_indices]
                triplet_targets = [triplet_targets[i] for i in sampled_indices]

            if batch_triplets:
                triplet_logits = triplet_model(batch_triplets)
                triplet_targets = torch.tensor(triplet_targets, dtype=torch.long, device=device)
            else:
                triplet_logits = torch.empty(0, 8).to(device)
                triplet_targets = torch.empty(0, dtype=torch.long, device=device)

            loss, mention_loss, triplet_loss = compute_loss(mention_logits, mention_targets, triplet_logits, triplet_targets, use_focal=True, use_focal_for_triplet=True)

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

            # Memory cleanup
            del span_embeddings, mention_logits, mention_probs, batch_triplets, triplet_logits, triplet_targets, mention_targets
            torch.cuda.empty_cache()
            gc.collect()

        epoch_msg = f"Epoch {epoch+1} / {epochs} | Loss: {total_loss:.4f} | Mention Loss: {total_mention_loss:.4f} | Triplet Loss: {total_triplet_loss:.4f}"
        print(epoch_msg)

        if val_loader is not None:
            print("\nRunning validation...")
            with torch.no_grad():
                output_metrics = evaluate(val_loader, encoder, mention_model, triplet_model)

        mention_scheduler.step(output_metrics['macro_f1_mention'])
        triplet_scheduler.step(output_metrics['macro_f1_triplet'])
        encoder_scheduler.step(output_metrics['micro_f1_triplet']['f1'])
        current_encoder_lr = mention_scheduler.optimizer.param_groups[0]['lr']
        current_mention_lr = mention_scheduler.optimizer.param_groups[1]['lr']
        current_triplet_lr = mention_scheduler.optimizer.param_groups[2]['lr']
        print(f"[Scheduler] Current LR Encoder after Epoch {epoch+1}: {current_encoder_lr:.2e}")
        print(f"[Scheduler] Current LR Mention after Epoch {epoch+1}: {current_mention_lr:.2e}")
        print(f"[Scheduler] Current LR Triplet after Epoch {epoch+1}: {current_triplet_lr:.2e}")

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
