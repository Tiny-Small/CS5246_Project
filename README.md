# CS5246 Project: Mining Aspect–Emotion Signals from Social Media via Span-Based Approach

> **TL;DR**: A modular pipeline for extracting aspect–emotion pairs from Facebook and Reddit comments using span-based models. Includes full and simplified ASTE implementations + GPT-powered preprocessing.

🧩 This repo covers my part of the CS5246 group project — mainly the span-based models for aspect–emotion extraction, along with data preprocessing scripts.

Other parts of the project, like data scraping, NER, or emotion classifier baselines, were handled separately and aren’t included here.

A modular pipeline for extracting and classifying emotion-labeled entities from public social media comments. The repository features three main components:

- **preprocessing folder**: Scripts and LLM prompts for cleaning, chunking, and emotion/entity annotation of raw Facebook and Reddit comments.

- **Span-ASTE with Modified Pruning (span_aste_modified pruning folder)**: An implementation of the full span-level ASTE model (Xu et al., 2021), adapted with simplified pruning via negative and random sampling.

- **Span-ASTE without opinions (span_aste folder)**: A streamlined variant that skips opinion extraction and directly classifies (aspect, emotion) pairs — better suited for noisy, informal social text.

## Repository Structure

| Folder | Description |
|--------|-------------|
| `preprocessing/` | Data cleaning, GPT annotation scripts, slang normalization, and manual vetting routines. |
| `span_aste_modified_pruning/` | Full ASTE pipeline with span classification, simplified pruning, and triplet prediction. |
| `span_aste/` | Simplified ASTE without opinion spans; uses a leaner architecture for robust emotion extraction. |

## How to Use

Each folder contains its own `README.md` with setup instructions, training scripts, and parameters explanations.

```bash
cd preprocessing/
# Run data cleaning or GPT annotation

cd span_aste/
# Train or evaluate modified ASTE (no opinions)

cd span_aste_modified_pruning/
# Train or evaluate full ASTE with pruning
```
