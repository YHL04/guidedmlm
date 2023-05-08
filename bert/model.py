

import torch.nn as nn

from .transformer import Transformer


class BERT(nn.Module):
    """
    BERT
    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 n_layers,
                 d_model,
                 n_head,
                 p
                 ):
        super(BERT, self).__init__()

        self.transformer = Transformer(
            vocab_size=vocab_size,
            max_len=max_len,
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
