from typing import Dict

import torch
import torch.nn as nn


class SeqClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        recurrent_struc: str
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embedding_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        if recurrent_struc == 'rnn':
            self.recurrent = nn.RNN(self.embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        elif recurrent_struc == 'lstm':
            self.recurrent = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        elif recurrent_struc == 'gru':
            self.recurrent = nn.GRU(self.embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        else:
            raise ValueError(f"Select a recurrent structure within rnn, lstm, or gru.")
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * (1 + self.bidirectional), num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size * (1 + self.bidirectional)

    def forward(self, batch, _len) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # reference: https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
        embedded = self.embedding(batch)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, _len, batch_first=True)
        if isinstance(self.recurrent, nn.LSTM):
            packed_output, (hidden, cell) = self.recurrent(packed_embedded)
        else:
            packed_output, hidden = self.recurrent(packed_embedded)
        if self.bidirectional:
            hidden = torch.cat((hidden[-1], hidden[-2]), dim=-1)
        else:
            hidden = hidden[-1]
        output = self.fc(hidden)
        return output


class SlotTagger(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        recurrent_struc: str
    ) -> None:
        super(SlotTagger, self).__init__()
        self.embedding_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        if recurrent_struc == 'rnn':
            self.recurrent = nn.RNN(self.embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        elif recurrent_struc == 'lstm':
            self.recurrent = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        elif recurrent_struc == 'gru':
            self.recurrent = nn.GRU(self.embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        else:
            raise ValueError(f"Select a recurrent structure within rnn, lstm, or gru.")
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * (1 + self.bidirectional), num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        return self.hidden_size * (1 + self.bidirectional)

    def forward(self, batch, _len) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(batch)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, _len, batch_first=True)
        if isinstance(self.recurrent, nn.LSTM):
            packed_output, (hidden, cell) = self.recurrent(packed_embedded)
        else:
            packed_output, hidden = self.recurrent(packed_embedded)
        unpacked_output, unpacked_lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(unpacked_output)
        return output
