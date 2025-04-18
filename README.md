# CS5246 Text Mining Group Project (Team 14): Emotion Analysis of Social Media Comments

> **TL;DR**: A modular pipeline for emotion classification, named entity recognition, and extracting aspect–emotion pairs using span-based models from Facebook and Reddit comments. Includes full and simplified ASTE implementations, GPT-powered preprocessing, NER, and emotion classification baselines.

🧩 This repo includes the complete modeling pipeline for extracting and classifying emotion-labeled entities from user-generated social media comments.

The repository features five main components:

- **preprocessing folder**: Scripts and prompts for cleaning, chunking, and annotating raw Facebook and Reddit comments using GPT-4o-mini

- **NER (Named Entity Recognition)**: Baselines using pretrained and fine-tuned models to detect entities.

- **Emotion Classifier**: A BERT-based masked/unmasked classifier for predicting emotions towards extracted entities.

- **Span-ASTE with Modified Pruning (span_aste_modified pruning folder)**: An implementation of the full ASTE framework (Xu et al., 2021) with simplified pruning via negative and random sampling.

- **Span-ASTE without opinions (span_aste folder)**: A streamlined ASTE variant that skips opinion spans and focuses on (aspect, emotion) pairs.

⚠️ **Note:** Some data files may have undergone minor column header adjustments during development. If errors occur when running the scripts, please verify and adjust the data headers accordingly.

## Repository Structure

| Folder | Description |
|--------|-------------|
| `scraping` | Scripts for scraping Reddit comments used in the dataset. |
| `preprocessing/` | Data cleaning, GPT annotation scripts, slang normalization, and manual vetting routines. |
| `data` | Contains the datasets used in our evaluations, including raw, cleaned, and annotated versions. |
| `ner/` | Codes for pretrained (e.g., spaCy, BERT-NER) and fine-tuned NER models. |
| `bert_emotion_classifier/` | BERT-based masked/unmasked emotion classification using entity-aware inputs. |
| `span_aste_modified_pruning/` | Full ASTE pipeline with span classification, simplified pruning, and triplet prediction. |
| `span_aste/` | Simplified ASTE without opinion spans; optimized for noisy, informal input. |

## How to Use

Each folder contains its own `README.md` with setup instructions, training scripts, and parameters explanations.

```bash
cd scraping/
# Scrape Reddit comments

cd preprocessing/
# Clean, segment, and annotate comments using GPT and heuristic rules

cd ner/
# Evaluate or fine-tune NER models (spaCy, BERT-NER)

cd emotion_classifier/
# Train or evaluate masked/unmasked BERT-based emotion classifiers

cd span_aste_modified_pruning/
# Train or evaluate full ASTE model with simplified pruning strategy

cd span_aste/
# Train or evaluate ASTE model (no opinions, simplified pruning)
```
