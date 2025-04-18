import os
import logging
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from random import randint, shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer

"""# Data Ingestion & Dataset Class"""

class SocialMediaDS(Dataset):
    def __init__(self, data_path,
                 sheet_name,
                 model_ckpt,
                 device,
                 max_token_length=50):
        super().__init__()
        self.device = device
        self.max_token_length = max_token_length

        # label encodings
        self.emotion_labels = [
            'ANGER',
            'DISGUST',
            'FEAR',
            'JOY',
            'NEUTRAL',
            'SADNESS',
            'SURPRISE'
        ]

        # load excel
        self.data = pd.read_excel(data_path, sheet_name=sheet_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ENTITY]"]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      row = self.data.loc[idx, :]

      entity = row['entity'].strip()
      comment = row['text']

      # Split comment into words while preserving the original entity
      comment_words = comment.split()

      # Find the entity's position and length in the word list
      entity_words = entity.split()
      entity_word_count = len(entity_words)

      # Replace the entity with [ENTITY] marker
      entity_start_idx = -1
      for i in range(len(comment_words) - entity_word_count + 1):
        if ' '.join(comment_words[i:i + entity_word_count]) == entity:
          entity_start_idx = i
          # Replace the entity span with single [ENTITY] token
          comment_words = (comment_words[:i] + ["[ENTITY]"] +
                           comment_words[i + entity_word_count:])
          break

      # Handle long comments
      if len(comment_words) > self.max_token_length:
        # Calculate window around entity
        half_window = int(self.max_token_length / 2)

        # Adjust start and end to keep entity in view
        start = max(entity_start_idx - half_window, 0)
        end = start + self.max_token_length

        # Ensure we don't exceed list bounds
        if end > len(comment_words):
            end = len(comment_words)
            start = max(0, end - self.max_token_length)

        comment_words = comment_words[start:end]

      # Join words back together and clean up
      comment = ' '.join(comment_words)
      comment = comment.replace("\\", "")

      emotion = row['emotion']
      emotion = self.emotion_labels.index(emotion)

      return f"{comment} [SEP] [ENTITY]", emotion

    def choose(self):
        return self[randint(0, len(self)-1)]

    def get_tokenizer_size(self):
        return len(self.tokenizer)

    def decode(self, input_id):
        return self.tokenizer.decode(input_id)

    def collate_fn(self, data):
        comments, emotions = zip(*data)
        comments = self.tokenizer(comments,
                                  padding=True,
                                  return_tensors='pt')
        comments = {k:v.to(self.device) for k, v in comments.items()}
        emotions = torch.tensor(emotions).long().to(self.device)
        return comments, emotions

class EmotionClassifier(nn.Module):
    def __init__(self, model_ckpt,
                 emotion_nlabels=1,
                 tokenizer_size=30523,
                 dropout_prob=0.35):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_ckpt)
        self.encoder.resize_token_embeddings(tokenizer_size)
        encoder_config = self.encoder.config
        self.emotion_classifier = nn.Sequential(
            nn.BatchNorm1d(encoder_config.hidden_size),
            nn.Dropout(dropout_prob),
            nn.Linear(encoder_config.hidden_size, emotion_nlabels)
        )

        self.emotion_classifier.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)

    def get_summary(self):
        print(self)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.encoder(**x)
        x = x.last_hidden_state[:, 0] # [cls] emb
        emotion_outputs = self.emotion_classifier(x)
        return emotion_outputs

"""## Non-Masked Dataset Class"""

class SocialMediaDS_NonMasked(SocialMediaDS):
    def __init__(self, data_path, sheet_name, model_ckpt, device, max_token_length=50):
        super().__init__(data_path, sheet_name, model_ckpt, device, max_token_length)
        
        # Re-initialize tokenizer without special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
    def __getitem__(self, idx):
      row = self.data.loc[idx, :]

      entity = row['entity'].strip()
      comment = row['text']

      # Split comment into words while preserving the original entity
      comment_words = comment.split()

      # Find the entity's position and length in the word list
      entity_words = entity.split()
      entity_word_count = len(entity_words)

      # Replace the entity with [ENTITY] marker
      entity_start_idx = -1
      for i in range(len(comment_words) - entity_word_count + 1):
        if ' '.join(comment_words[i:i + entity_word_count]) == entity:
          entity_start_idx = i
          break

      # Handle long comments
      if len(comment_words) > self.max_token_length:
        # Calculate window around entity
        half_window = int(self.max_token_length / 2)

        # Adjust start and end to keep entity in view
        start = max(entity_start_idx - half_window, 0)
        end = start + self.max_token_length

        # Ensure we don't exceed list bounds
        if end > len(comment_words):
            end = len(comment_words)
            start = max(0, end - self.max_token_length)

        comment_words = comment_words[start:end]

      # Join words back together and clean up
      comment = ' '.join(comment_words)
      comment = comment.replace("\\", "")

      emotion = row['emotion']
      emotion = self.emotion_labels.index(emotion)

      return f"{comment} [SEP] {entity}", emotion

"""# Model Training & Validation"""

def train(data, model, optimizer, emo_loss_fn):
    model.train()
    comments, emotions = data
    emo_outputs = model(comments)
    emo_loss = emo_loss_fn(emo_outputs, emotions)
    loss = emo_loss
    model.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
    optimizer.step()

    emo_preds = emo_outputs.argmax(-1)
    emo_metrics = compute_metrics(emotions, emo_preds)
    return loss, emo_metrics

@torch.no_grad()
def validate(data, model, emo_loss_fn):
    model.eval()
    comments, emotions = data
    emo_outputs = model(comments)
    emo_loss = emo_loss_fn(emo_outputs, emotions)
    loss = emo_loss

    emo_preds = emo_outputs.argmax(-1)
    emo_metrics = compute_metrics(emotions, emo_preds)
    return loss, emo_metrics

def train_and_validate(model, train_dl, val_dl, loss_fn, optimizer, scheduler, model_save_path, epochs=5):
  mem = {
      'train_loss': [],
      'train_acc': [],
      'train_f1': [],
      'val_loss': [],
      'val_acc': [],
      'val_f1': []
  }

  cur_best_f1 = 0

  for epoch in tqdm(range(epochs), desc='Training'):

      n_batch = len(train_dl)
      train_losses = []
      train_accs = []
      train_f1s = []

      for i, data in enumerate(train_dl):
          train_loss, train_metrics = train(data, model, optimizer, loss_fn)
          pos = epoch + ((i+1)/n_batch)
          train_losses.append(train_loss.cpu().detach())
          train_accs.append(train_metrics['acc'])
          train_f1s.append(train_metrics['f1'])

      mem['train_loss'].append(np.mean(train_losses))
      mem['train_acc'].append(np.mean(train_accs))
      mem['train_f1'].append(np.mean(train_f1s))

      n_batch = len(val_dl)
      val_losses = []
      val_accs = []
      val_f1s = []

      for i, data in enumerate(val_dl):
          val_loss, val_metrics = validate(data, model, loss_fn)
          pos = epoch + ((i+1)/n_batch)
          val_losses.append(val_loss.cpu().detach())
          val_accs.append(val_metrics['acc'])
          val_f1s.append(val_metrics['f1'])
          msg = f"epoch: {pos:.3f}\tval loss: {val_loss:.3f}\tval_acc: {val_metrics['acc']:.3f}\tval_f1: {val_metrics['f1']:.3f}"

      mem['val_loss'].append(np.mean(val_losses))
      mem['val_acc'].append(np.mean(val_accs))
      mem['val_f1'].append(np.mean(val_f1s))

      # Logging
      log = (f"Epoch {epoch+1}/{epochs}\n"
            f"Train Loss: {mem['train_loss'][-1]:.3f}\tAcc: {mem['train_acc'][-1]:.3f}\tF1: {mem['train_f1'][-1]:.3f}\n"
            f"Val Loss:   {mem['val_loss'][-1]:.3f}\tAcc:   {mem['val_acc'][-1]:.3f}\tF1:   {mem['val_f1'][-1]:.3f}\n")
      tqdm.write(log)

      scheduler.step()

      # Save best model based on val_f1
      if mem['val_f1'][-1] > cur_best_f1:
          torch.save(model.state_dict(), model_save_path)
          cur_best_f1 = mem['val_f1'][-1]
          tqdm.write(f"New best F1: {cur_best_f1:.3f} - Model saved!")

  print("Training completed!")
  return mem

"""# Evaluation & Metrics"""

def compute_metrics(targets, preds):
    targets = targets.cpu().detach()
    preds = preds.cpu().detach()
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    return {'acc': acc, 'f1': f1, 'preds': preds, 'targets':targets}

def focal_loss(gamma=2):
    def compute_loss(preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1-pt)**gamma * ce_loss).mean()
    return compute_loss

def evaluate_model(model, val_ds, val_dl):
    emo_preds, emo_targets, all_texts = [], [], []

    tokenizer = val_ds.tokenizer

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dl), total=len(val_dl)):
            comments, emotions = data
            emo_outputs = model(comments)

            emo_preds.extend(emo_outputs.argmax(-1).cpu().numpy())
            emo_targets.extend(emotions.cpu().numpy())

            decoded_texts = tokenizer.batch_decode(comments["input_ids"], skip_special_tokens=True)
            all_texts.extend(decoded_texts)

    report_dict = classification_report(emo_targets, emo_preds, digits=4, output_dict=True)

    print(f"{'':<15}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    
    for label in report_dict:
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        row = report_dict[label]
        print(f"{str(label):<15}{row['precision']*100:10.2f}{row['recall']*100:10.2f}{row['f1-score']*100:10.2f}{row['support']:10.0f}")

    print(f"\n{'accuracy':<35}{report_dict['accuracy']*100:10.2f}{report_dict['macro avg']['support']:10.0f}")

    for avg in ['macro avg', 'weighted avg']:
        row = report_dict[avg]
        print(f"{avg:<15}{row['precision']*100:10.2f}{row['recall']*100:10.2f}{row['f1-score']*100:10.2f}{row['support']:10.0f}")

    return emo_preds, emo_targets, all_texts

def plot_confusion_matrix(emo_preds, emo_targets):
    plt.rcParams['figure.figsize'] = [10, 10]

    all_emotions = ['ANGER', 'DISGUST', 'FEAR', 'JOY', 'NEUTRAL', 'SADNESS', 'SURPRISE']
    unique_classes = sorted(set(emo_targets) | set(emo_preds))
    labels_to_display = [all_emotions[i] for i in unique_classes]

    ConfusionMatrixDisplay.from_predictions(
        emo_targets,
        emo_preds,
        cmap='flare',
        display_labels=labels_to_display
    )
    plt.title("Confusion Matrix")
    plt.show()

"""# Error Analysis & Visualization"""

def analyze_misclassifications(emo_preds, emo_targets, all_texts):
    emotion_labels = ['ANGER', 'DISGUST', 'FEAR', 'JOY', 'NEUTRAL', 'SADNESS', 'SURPRISE']

    # Ensure inputs are NumPy arrays
    emo_preds = np.array(emo_preds)
    emo_targets = np.array(emo_targets)

    # Create a DataFrame for analysis
    misclassified_df = pd.DataFrame({
        "Text": np.array(all_texts, dtype=object),
        "True Emotion": [emotion_labels[i] for i in emo_targets],
        "Predicted Emotion": [emotion_labels[i] for i in emo_preds]
    })

    # Define a severity function
    def classify_severity(true_label, predicted_label):
        """ Defines severity based on sentiment shifts (e.g., negative to positive) """
        negative_emotions = {"ANGER", "DISGUST", "FEAR", "SADNESS"}
        positive_emotions = {"JOY", "SURPRISE"}
        neutral_emotions = {"NEUTRAL"}

        if (true_label in negative_emotions and predicted_label in positive_emotions) or \
           (true_label in positive_emotions and predicted_label in negative_emotions):
            return "Severe Misclassification"
        elif true_label != predicted_label:
            return "Moderate Misclassification"
        else:
            return "Correct"

    # Apply severity classification
    misclassified_df["Severity"] = misclassified_df.apply(
        lambda row: classify_severity(row["True Emotion"], row["Predicted Emotion"]),
        axis=1
    )

    return misclassified_df

def plot_severity_bar(misclassified_df, exclude_neutral=True, show_percent=False):
    df_grouped = misclassified_df.groupby(["True Emotion", "Severity"]).size().reset_index(name="Count")

    if show_percent:
        df_total = df_grouped.groupby("True Emotion")["Count"].transform("sum")
        df_grouped["Value"] = df_grouped["Count"] / df_total * 100
        y_label = "Percentage"
        label_fmt = "%.1f%%"
    else:
        df_grouped["Value"] = df_grouped["Count"]
        y_label = "Count"
        label_fmt = "%d"

    if exclude_neutral:
        df_grouped = df_grouped[df_grouped["True Emotion"] != "NEUTRAL"]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_grouped, x="True Emotion", y="Value", hue="Severity")
    plt.title(f"Classification Outcome by Emotion ({y_label})")
    plt.ylabel(y_label)
    plt.xticks(rotation=45)

    # Add value labels on top
    for container in ax.containers:
        ax.bar_label(container, fmt=label_fmt, label_type="edge", fontsize=9, padding=2)

    plt.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()