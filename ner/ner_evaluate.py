import spacy
from typing import List, Tuple
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

def evaluate_spacy(docs: List[Tuple[str, List[str]]]):
    nlp = spacy.load("en_core_web_trf")
    TP = 0
    FP = 0
    tpfp_count = 0
    count = 0
    for i in tqdm(range(len(docs))):
        doc = nlp(docs[i][0])
        entities = set([ent.text for ent in doc.ents])
        # gt = set(eval(docs[i][1]))
        gt = set(docs[i][1])

        TP += len(entities & gt)
        FP += len(entities - gt)
        tpfp_count = tpfp_count + TP + FP
        count += len(gt)
    precision = TP / (TP + FP)
    recall = TP / count
    return precision, recall

def evaluate_bert(docs: List[Tuple[str, List[str]]]):
    model_name = "AventIQ-AI/bert-named-entity-recognition"
    model = BertForTokenClassification.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    label_list = ["0", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    def predict_entities(text, model):
        tokens = tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**tokens)
        
        logits = outputs.logits  # Extract logits
        predictions = torch.argmax(logits, dim=2)  # Get highest probability labels

        tokens_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        predicted_labels = [label_list[pred] for pred in predictions[0].cpu().numpy()]

        final_tokens = []
        final_labels = []
        for token, label in zip(tokens_list, predicted_labels):
            if token.startswith("##"):  
                final_tokens[-1] += token[2:]  # Merge subword
            else:
                final_tokens.append(token)
                final_labels.append(label)

        entities = []
        labels = []
        for token, label in zip(final_tokens, final_labels):
            if token not in ["[CLS]", "[SEP]"] and label != "0":
                entities.append(token)
                labels.append(label)
        
        return entities, labels

    TP = 0
    FP = 0
    count = 0
    tpfp_count = 0

    for i in tqdm(range(len(docs))):
        entities, labels = predict_entities(docs[i][0], model)
        entities = set(entities)

        # print(docs[i][1])
        # gt = set(eval(docs[i][1]))
        gt = set(docs[i][1])

        TP += len(entities & gt)
        FP += len(entities - gt)
        tpfp_count = tpfp_count + TP + FP
        count += len(gt)
    precision = TP / (TP + FP)
    recall = TP / count
    return precision, recall


if __name__ == "__main__":
    import pandas as pd

    # df = pd.read_csv("data/train_grouped.csv")
    # df = pd.read_csv("data/Dataset 2_final_df6_train.csv")
    df = pd.read_csv("data/Dataset 3_final_df6B_train.csv")
    dataset = []
    for idx, row in df.iterrows():
        # dataset.append([row['text'], row['entity']])s
        dataset.append([row['text'], row['aspect']])
    
    print(evaluate_spacy(dataset))
    # print(evaluate_bert(dataset))