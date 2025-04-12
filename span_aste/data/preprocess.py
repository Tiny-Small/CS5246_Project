import torch
import ast

# import config
# from utils import get_mention_labels

import span_aste.config as config
from span_aste.utils import get_mention_labels


# Utility to generate all valid spans up to max length
def enumerate_spans(sentence_len, max_span_len=10):
    return [(i, j) for i in range(sentence_len) for j in range(i, min(i + max_span_len, sentence_len))]

def preprocess_row(row, tokenizer, max_span_len=10, filtered_spans=True, aspect_sentiment_only=False):

    text = row['text']
    annotations = row['annotations']  # list of dicts with aspect, opinion, emotion

    tokens = tokenizer.tokenize(text.lower())

    if filtered_spans and "filtered_spans" in row:
        spans = ast.literal_eval(row["filtered_spans"])
    else:
        spans = enumerate_spans(len(tokens), max_span_len)

    # Build index of all annotated target/opinion spans
    target_spans = set()
    opinion_spans = set()
    if aspect_sentiment_only:
        labeled_aspects = {}
    else:
        labeled_triplets = {}

    def find_span(entity):
        entity_tokens = tokenizer.tokenize(entity.lower())
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                return i, i + len(entity_tokens) - 1
        print(f"[SPAN NOT FOUND] Entity: \"{entity}\" | Tokens: {tokens}")
        return None

    for ann in annotations:
        aspect_span = find_span(ann['aspect'])
        emotion = ann['emotion'].upper()
        label = config.EMOTION_LABELS.get(emotion, config.EMOTION_LABELS['INVALID'])

        if aspect_span is not None:
            target_spans.add(aspect_span)

        if aspect_sentiment_only:
            if aspect_span:
                labeled_aspects[aspect_span] = label
        else:
            opinion_span = find_span(ann['opinion'])
            if opinion_span:
                opinion_spans.add(opinion_span)
            if aspect_span and opinion_span:
                labeled_triplets[(aspect_span, opinion_span)] = label

    mention_label_map = get_mention_labels(aspect_sentiment_only)

    mention_labels = []
    if aspect_sentiment_only:
        for span in spans:
            if span in target_spans:
                mention_labels.append(mention_label_map["TARGET"])
            else:
                mention_labels.append(mention_label_map["INVALID"])
    else:
        for span in spans:
            if span in target_spans:
                mention_labels.append(mention_label_map["TARGET"])
            elif span in opinion_spans:
                mention_labels.append(mention_label_map["OPINION"])
            else:
                mention_labels.append(mention_label_map["INVALID"])

    if aspect_sentiment_only:
        return {
            'text': text,
            'tokens': tokens,
            'spans': spans,
            'mention_labels': mention_labels,
            'aspect_labels': labeled_aspects
        }
    else:
        triplet_labels = []
        for t in spans:
            for o in spans:
                if t == o: continue
                label = labeled_triplets.get((t, o), config.EMOTION_LABELS['INVALID'])
                triplet_labels.append((t, o, label))

        return {
            'text': text,
            'tokens': tokens,
            'spans': spans,
            'mention_labels': mention_labels,
            'triplet_labels': triplet_labels
        }

def group_rows_by_text_df(df, aspect_sentiment_only=False):
    if aspect_sentiment_only:
        grouped = (
            df.groupby('text', group_keys=False)
            .apply(lambda g: {
                'text': g.name,
                'annotations': g[['aspect', 'emotion']].to_dict(orient='records')
            }, include_groups=False)
            .tolist()
        )
    else:
        grouped = (
            df.groupby('text', group_keys=False)
            .apply(lambda g: {
                'text': g.name,
                'annotations': g[['aspect', 'opinion', 'emotion']].to_dict(orient='records')
            }, include_groups=False)
            .tolist()
        )
    return grouped
