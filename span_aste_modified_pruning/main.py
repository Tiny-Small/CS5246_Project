import argparse
import pandas as pd
import numpy as np
import torch

from span_aste.train import train
from span_aste.evaluate import evaluate
from span_aste.models.encoder import SentenceEncoder
from span_aste.models.mention import MentionClassifier
from span_aste.models.triplet import TripletClassifier
from span_aste.data.dataset import setup_dataloader
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_span_len', type=int, default=5)
    parser.add_argument('--checkpoint_path', type=str, default="/path/CS5246/span_aste/checkpoints")
    parser.add_argument('--checkpoint_filepath', type=str, default="/path/CS5246/span_aste/checkpoints/2/checkpoint_epoch1.pt")
    parser.add_argument('--log_filepath', type=str, default="logs/train.log")
    args = parser.parse_args()

    #df_train_val = pd.read_excel(args.data_path)
    # max_words = int(np.percentile(df_train_val['word_count'], 90))
    # df_train_val = df_train_val[df_train_val["word_count"] <= max_words]
    #df_train_val = df_train_val[~df_train_val['emotion'].isin(["SURPRISE", "FEAR", "ANGER"])]

    #df = pd.DataFrame(
    #    {
    #        "text": df_train_val.text,
    #        "aspect": df_train_val.entity,
    #        #"aspect": df_train_val.aspect,
    #        "opinion": df_train_val.opinion,
    #        "emotion": df_train_val.emotion,
    #        "filtered_spans": df_train_val.filtered_spans
    #    }
    #)
    #train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    #val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['emotion'])

    #train_df =  pd.read_excel("/path/CS5246/data/Res14_train_df.xlsx")
    #val_df =  pd.read_excel("/path/CS5246/data/Res14_val_df.xlsx")
    #test_df =  pd.read_excel("/path/CS5246/data/Res14_test_df.xlsx")

    train_df =  pd.read_excel("/path/CS5246/data/final_df6B_train.xlsx")
    val_df =  pd.read_excel("/path/CS5246/data/final_df6B_val.xlsx")
    test_df =  pd.read_excel("/path/CS5246/data/final_df6B_test.xlsx")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SentenceEncoder().to(device)
    embedding_dim = encoder.bert.config.hidden_size

    mention_model = MentionClassifier(input_dim=embedding_dim).to(device)
    triplet_model = TripletClassifier(input_dim=embedding_dim, num_classes=4).to(device)

    if args.mode == 'train':
        train(df=train_df, val_df=val_df, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_span_len=args.max_span_len, save_dir=args.checkpoint_path, log_path=args.log_filepath)

        # Load the last saved checkpoint for evaluation
        checkpoint_path = f"{args.checkpoint_path}/checkpoint_epoch{args.epochs}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        mention_model.load_state_dict(checkpoint['mention_model'])
        triplet_model.load_state_dict(checkpoint['triplet_model'])

        dataloader, _ = setup_dataloader(test_df, tokenizer=encoder.tokenizer, batch_size=args.batch_size)
        print("\nEvaluating on test set:")
        _ = evaluate(dataloader, encoder, mention_model, triplet_model)

    elif args.mode == 'eval':
        checkpoint = torch.load(args.checkpoint_filepath, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        mention_model.load_state_dict(checkpoint['mention_model'])
        triplet_model.load_state_dict(checkpoint['triplet_model'])

        dataloader, _ = setup_dataloader(test_df, tokenizer=encoder.tokenizer, batch_size=args.batch_size)
        print("\nEvaluating on test dataset:")
        _ = evaluate(dataloader, encoder, mention_model, triplet_model)

if __name__ == '__main__':
    main()
