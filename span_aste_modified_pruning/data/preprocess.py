import torch
import ast
from span_aste.utils import EMOTION_LABELS

# Utility to generate all valid spans up to max length
def enumerate_spans(sentence_len, max_span_len=10):
    return [(i, j) for i in range(sentence_len) for j in range(i, min(i + max_span_len, sentence_len))]

def preprocess_row(row, tokenizer, max_span_len=10, filtered_spans=True):
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
    labeled_triplets = {}

    def find_span(entity):
        entity_tokens = tokenizer.tokenize(entity.lower())
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                return i, i + len(entity_tokens) - 1
        return None

    for ann in annotations:
        aspect_span = find_span(ann['aspect'])
        opinion_span = find_span(ann['opinion'])

        if aspect_span is None:
            print(f"[MISSING ASPECT] \"{ann['aspect']}\" in: \"{text}\"")
        if opinion_span is None:
            print(f"[MISSING OPINION] \"{ann['opinion']}\" in: \"{text}\"")

        emotion = ann['emotion'].upper()
        label = EMOTION_LABELS.get(emotion, EMOTION_LABELS['INVALID'])
        if aspect_span: target_spans.add(aspect_span)
        if opinion_span: opinion_spans.add(opinion_span)
        if aspect_span and opinion_span:
            labeled_triplets[(aspect_span, opinion_span)] = label

    mention_labels = []
    for span in spans:
        if span in target_spans:
            mention_labels.append(0)
        elif span in opinion_spans:
            mention_labels.append(1)
        else:
            mention_labels.append(2)

    triplet_labels = []
    for t in spans:
        for o in spans:
            if t == o: continue
            label = labeled_triplets.get((t, o), EMOTION_LABELS['INVALID'])
            triplet_labels.append((t, o, label))

    return {
        'text': text,
        'tokens': tokens,
        'spans': spans,
        'mention_labels': mention_labels,
        'triplet_labels': triplet_labels
    }

def group_rows_by_text_df(df):
    grouped = (
        df.groupby('text', group_keys=False)
          .apply(lambda g: {
              'text': g.name,
              'annotations': g[['aspect', 'opinion', 'emotion']].to_dict(orient='records')
          }, include_groups=False)
          .tolist()
    )
    return grouped
