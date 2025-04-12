# Span-ASTE (Modified Pruning)

This directory contains an implementation based on Xu et al. (2021), with a simplified pruning strategy using negative and random sampling. The model performs end-to-end extraction of (aspect, opinion, emotion) triplets from user comments

## Overview

The ASTE model:
- Enumerates all possible contiguous spans and classifies them as **Aspect**, **Opinion**, or **Invalid** using a feedforward neural network (the mention module).
- Forms candidate (aspect, opinion) pairs and feeds them into a second feedforward network (the *triplet module*) to classify the associated emotion.
- Using the distance embedding to capture the relative position between aspect and opinion spans.
- **Prunes** the $O(n^4)$ span pair combinations by randomly sampling a fixed number of span pairs --- this differs from the original dual-channel pruning strategy used by Xu et al.
- Uses **focal loss** instead of standard cross-entropy to better handle class imbalance.

## Requirements

- Python 3.10.12

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running
To run the scripts:
```bash
python -m span_aste.main --mode eval --data_path /path/CS5246/data/final_df6.xlsx --epochs 1 --batch_size 4 --max_span_len 8 --checkpoint_filepath /path/CS5246/span_aste/checkpoints/20/checkpoint_epoch82.pt --checkpoint_path /path/CS5246/span_aste/checkpoints/20 --log_filepath /path/CS5246/span_aste/logs/train20_test.log
```
where:

| Parameter | Explanation |
|-----------|-------------|
| `--mode eval` | Mode of execution: `train`, or `eval`. |
| `--data_path ` | Path to the dataset file. |
| `--epochs` | Number of epochs to train. Set to 1 for debugging. |
| `--batch_size` | Number of samples per batch. |
| `--max_span_len` | Maximum span length to consider as aspect or opinion. |
| `--checkpoint_filepath` | Path to a specific checkpoint to resume from. Leave blank to start fresh. |
| `--checkpoint_path` |	Directory to save or load checkpoints. |
| `--log_filepath` | Path to store training or evaluation logs. |
