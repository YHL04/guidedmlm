

import torch
import torch.nn as nn

from .transformer import Transformer


class BERT(nn.Module):
    """
    BERT
    """

    def __init__(self,
                 vocab_size,
                 n_layers=4,
                 d_model=128,
                 n_head=8,
                 p=0.1
                 ):

        super(BERT, self).__init__()

        self.transformer = Transformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, ids):
        x = self.transformer.forward(ids)
        x = self.softmax(self.linear(x))

        return x
