import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTagDataset
from model import SlotTagger
from utils import Vocab

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from focal_loss import FocalLoss

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    # print(tag2idx)

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTagDataset] = {
        split: SeqTagDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # print(datasets["train"][0])
    
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=datasets["train"].collate_fn, num_workers=8)
    eval_loader = DataLoader(datasets["eval"], batch_size=args.batch_size, shuffle=False, collate_fn=datasets["eval"].collate_fn, num_workers=8)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # print(embeddings.shape)
    
    model = SlotTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, len(tag2idx), args.recurrent_struc, args.out_channels, args.kernel_size).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = FocalLoss(gamma=2) if args.loss == 'focal' else torch.nn.CrossEntropyLoss()

    best_acc = 0.7

    for epoch in range(args.num_epoch):
        model.train()
        train_loss = 0
        hit = 0
        for data, _len in tqdm(train_loader):
            data['tokens'], data['tags'] = data['tokens'].to(args.device), data['tags'][:,:max(_len)].to(args.device)
            logit = model(data['tokens'], _len)
            loss = criterion(logit.permute(0, 2, 1), data['tags'])
            train_loss += loss.item()
            hit += sum([l == d for l, d in zip(logit.argmax(dim=-1).tolist(), data['tags'].tolist())])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        train_joint_acc = hit / len(datasets["train"])
        print(f"[ Train | {epoch + 1:03d}/{args.num_epoch:03d} ] loss = {train_loss:.5f}, joint acc = {train_joint_acc:.5f}")

        model.eval()
        hit = 0
        for data, _len in tqdm(eval_loader):
            data['tokens'], data['tags'] = data['tokens'].to(args.device), data['tags'][:,:max(_len)].to(args.device)
            with torch.no_grad():
                logit = model(data['tokens'], _len)
            hit += sum([l == d for l, d in zip(logit.argmax(dim=-1).tolist(), data['tags'].tolist())])
        eval_joint_acc = hit / len(datasets["eval"])
        print(f"[ Eval | {epoch + 1:03d}/{args.num_epoch:03d} ] joint acc = {eval_joint_acc:.5f}")

        if eval_joint_acc > best_acc:
            best_acc = eval_joint_acc
            print('Saving model with best joint accuracy {:.3f}'.format(best_acc))
            torch.save(model.state_dict(), f"./ckpt/slot/{args.ckpt_name}")

    print(f"Finish training. The saved weight, {args.ckpt_name}, has {best_acc} joint accuracy.")

            
        


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        help="Name of model checkpoint.",
        required=True
    )

    parser.add_argument("--max_len", type=int, default=128)

    parser.add_argument("--recurrent_struc", type=str, help="rnn, lstm, gru, cnnlstm", default="lstm")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    # for cnnlstm
    parser.add_argument("--out_channels", type=int, default=100)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    parser.add_argument("--loss", type=str, default="ce")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
