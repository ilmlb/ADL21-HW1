import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    # print(intent2idx)

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # print(datasets["train"][0])
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=datasets["train"].collate_fn, num_workers=8)
    eval_loader = DataLoader(datasets["eval"], batch_size=args.batch_size, shuffle=False, collate_fn=datasets["eval"].collate_fn, num_workers=8)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # print(embeddings.shape)
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, len(intent2idx), args.recurrent_struc).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.85

    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in range(args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss = 0
        hit = 0
        for data, _len in tqdm(train_loader):
            data['text'], data['intent'] = data['text'].to(args.device), data['intent'].to(args.device)
            logit = model(data['text'], _len)
            loss = criterion(logit, data['intent'])
            train_loss += loss.item()
            hit += (logit.argmax(dim=-1) == data['intent']).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        train_acc = hit / len(datasets["train"])
        print(f"[ Train | {epoch + 1:03d}/{args.num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        hit = 0
        for data, _len in tqdm(eval_loader):
            data['text'], data['intent'] = data['text'].to(args.device), data['intent'].to(args.device)
            with torch.no_grad():
                logit = model(data['text'], _len)
            hit += (logit.argmax(dim=-1) == data['intent']).sum().item()
        eval_acc = hit / len(datasets["eval"])
        print(f"[ Eval | {epoch + 1:03d}/{args.num_epoch:03d} ] acc = {eval_acc:.5f}")

        if eval_acc > best_acc:
            best_acc = eval_acc
            print('Saving model with best accuracy {:.3f}'.format(best_acc))
            torch.save(model.state_dict(), f"./ckpt/intent/{args.ckpt_name}")
    
    print(f"Finish training. The saved weight, {args.ckpt_name}, has {best_acc} accuracy.")

    # TODO: Inference on test set
        


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        help="Name of model checkpoint.",
        required=True
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--recurrent_struc", type=str, help="rnn, lstm, gru", default="lstm")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
