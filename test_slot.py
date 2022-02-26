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


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTagDataset(data, vocab, tag2idx, args.max_len)
    # print(dataset[0])
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SlotTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.recurrent_struc
    ).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    pred = []
    for data, _len in tqdm(test_loader):
        data['tokens'] = data['tokens'].to(args.device)
        with torch.no_grad():
            logit = model(data['tokens'], _len)
        idx = logit.argmax(dim=-1).cpu().tolist()
        for i, l in enumerate(_len):
            idx[i] = idx[i][:l]
        tags = [dataset.idx2tag(i) for i in idx]
        pred += zip(data['id'], tags)
    pred.sort(key=lambda x: int(x[0][5:]))
        

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, "w") as f:
        f.write("id,tags\n")
        for id, tag in  pred:
            f.write(f"{id},{' '.join(tag)}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--recurrent_struc", type=str, help="rnn, lstm, gru", default="lstm")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
