# Span-ASTE (Modified)

This directory contains the implementation of the **Modified Aspect Sentiment Triplet Extraction (ASTE)** model, which performs end-to-end extraction of (aspect, emotion) pairs from user comments.

## Overview

The modified ASTE model:
- Classifies all possible spans in a sentence as **Aspect** or **Invalid** using a feedforward network (mention module).
- Removes the opinion span identification present in the original ASTE, simplifying the architecture.
- Eliminates the need for pruning strategies since no aspect–opinion pairing is performed.
- Uses a second feedforward network (triplet module) to classify each aspect span into one of the predefined emotion categories.
- Is trained with **focal loss** to address class imbalance.

## Requirements

- Python 3.10.6

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running
To run the scripts:
```bash
python -m span_aste.main --mode train  --epochs 50 --batch_size 4 --max_span_len 5 --data_folder "/path/CS5246/data" --data_file "final_df6" --checkpoint_path "/path/CS5246/span_aste/checkpoints/Exp02C" --checkpoint_filepath "" --log_dir "/path/CS5246/span_aste/logs/Exp02C" --log_naming "Dataset2_final_df6_Exp02C" --gamma 2.0
```
where:

| Parameter | Explanation |
|-----------|-------------|
| `--mode train` | Sets the script mode. Options: `train`, `eval`. |
| `--epochs` | Number of training epochs. |
| `--batch_size` | Number of samples per batch. |
| `--max_span_len` | Maximum length of candidate spans (tokens) considered during classification |
| `--data_folder` | Directory containing the dataset file. |
| `--data_file`	| Dataset filename (without extension) inside `data_folder`. |
| `--checkpoint_path` | Directory to save model checkpoints. |
| `--checkpoint_filepath` | Path to a specific checkpoint file to resume from. |
| `--log_dir` | Directory for logs. |
| `--log_naming` | Naming convention for logs (e.g., `Test_01`). |
| `--gamma` | Gamma parameter for focal loss; controls weighting of hard vs. easy examples. |
