import argparse
import pandas as pd
import numpy as np
import torch
import json
import os
from sklearn.model_selection import train_test_split

# from train import train
# from evaluate import evaluate
# from models.encoder import SentenceEncoder
# from models.mention import MentionClassifier
# from models.triplet import TripletClassifier
# from utils import set_seed
# from data.dataset import setup_dataloader
# from config import init_emotion_labels
# import config

from span_aste.train import train
from span_aste.evaluate import evaluate
from span_aste.models.encoder import SentenceEncoder
from span_aste.models.mention import MentionClassifier
from span_aste.models.triplet import TripletClassifier
from span_aste.utils import set_seed
from span_aste.data.dataset import setup_dataloader
from span_aste.config import init_emotion_labels
import span_aste.config as config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_span_len', type=int, default=5)
    parser.add_argument('--aspect_sentiment_only', action='store_true') ##
    parser.add_argument('--data_folder', type=str, default="../data")
    parser.add_argument('--data_file', type=str, default="final_df6B")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints")
    parser.add_argument('--checkpoint_filepath', type=str, default="")
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--log_naming', type=str, default="v00")
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    init_emotion_labels(args.data_file)

    print(f"[DEBUG] Number of classes: {config.NUM_CLASSES}")

    set_seed(args.seed)

    train_df =  pd.read_excel(os.path.join(f"{args.data_folder}/{args.data_file}_train.xlsx"))
    val_df =  pd.read_excel(os.path.join(f"{args.data_folder}/{args.data_file}_val.xlsx"))
    test_df =  pd.read_excel(os.path.join(f"{args.data_folder}/{args.data_file}_test.xlsx"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SentenceEncoder().to(device)
    embedding_dim = encoder.bert.config.hidden_size

    if args.aspect_sentiment_only:
        mention_model = MentionClassifier(input_dim=embedding_dim, num_classes=2).to(device)
    else:
        mention_model = MentionClassifier(input_dim=embedding_dim, num_classes=3).to(device)

    triplet_model = TripletClassifier(input_dim=embedding_dim, num_classes=config.NUM_CLASSES, aspect_sentiment_only=args.aspect_sentiment_only).to(device) ##

    if args.mode == 'train':

        # Save args
        os.makedirs(args.log_dir, exist_ok=True)
        args_json_path = os.path.join(args.log_dir, f"{args.log_naming}_{args.mode}_args.json")
        with open(args_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[INFO] Saved training args to {args_json_path}")

        train(df=train_df, val_df=val_df,
              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_span_len=args.max_span_len,
              gamma=args.gamma, aspect_sentiment_only=args.aspect_sentiment_only,
              save_dir=args.checkpoint_path, log_dir=args.log_dir, log_naming=args.log_naming,
              load_checkpoint=args.checkpoint_filepath)

        # Load the last saved checkpoint for evaluation
        checkpoint_path = f"{args.checkpoint_path}/checkpoint_epoch{args.epochs}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        mention_model.load_state_dict(checkpoint['mention_model'])
        triplet_model.load_state_dict(checkpoint['triplet_model'])

        dataloader, _ = setup_dataloader(test_df, tokenizer=encoder.tokenizer, batch_size=args.batch_size,
                                         max_span_len=args.max_span_len, aspect_sentiment_only=args.aspect_sentiment_only)
        print("\nEvaluating on test set:")
        _ = evaluate(dataloader, encoder, mention_model, triplet_model, aspect_sentiment_only=args.aspect_sentiment_only) ##

    elif args.mode == 'eval':
        checkpoint = torch.load(args.checkpoint_filepath, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        mention_model.load_state_dict(checkpoint['mention_model'])
        triplet_model.load_state_dict(checkpoint['triplet_model'])

        dataloader, _ = setup_dataloader(test_df, tokenizer=encoder.tokenizer, batch_size=args.batch_size,
                                         max_span_len=args.max_span_len, aspect_sentiment_only=args.aspect_sentiment_only)
        print("\nEvaluating on test dataset:")
        _ = evaluate(dataloader, encoder, mention_model, triplet_model, aspect_sentiment_only=args.aspect_sentiment_only) ##


if __name__ == '__main__':
    main()
