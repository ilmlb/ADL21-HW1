from typing import List, Dict
import torch

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # reference: https://vodkazy.cn/2019/12/12/%E5%B0%8F%E8%AE%B0%EF%BC%9A%E5%A4%84%E7%90%86LSTM-embedding%E5%8F%98%E9%95%BF%E5%BA%8F%E5%88%97/
        samples.sort(key=lambda x: len(x['text'].split()), reverse=True)
        data_length = [len(sq['text'].split()) for sq in samples]
        return {
            'text': torch.tensor(self.vocab.encode_batch([sq['text'].split() for sq in samples], self.max_len), dtype=torch.int64),
            'intent': torch.tensor([self.label2idx(sq['intent']) for sq in samples], dtype=torch.int64) if 'intent' in samples[0] else None,
            'id': [sq['id'] for sq in samples]
        }, data_length
        

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        tag_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.tag_mapping = tag_mapping
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.tag_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        samples.sort(key=lambda x: len(x['tokens']), reverse=True)
        data_length = [len(sq['tokens']) for sq in samples]
        return {
            'tokens': torch.tensor(self.vocab.encode_batch([sq['tokens'] for sq in samples], self.max_len), dtype=torch.int64),
            'tags': torch.tensor(pad_to_len([self.tag2idx(sq['tags']) for sq in samples], self.max_len, 0), dtype=torch.int64) if 'tags' in samples[0] else None,
            'id': [sq['id'] for sq in samples]
        }, data_length
        

    def tag2idx(self, tag: List[str]):
        return [self.tag_mapping[t] for t in tag]

    def idx2tag(self, idx: List[int]):
        return [self._idx2tag[i] for i in idx]