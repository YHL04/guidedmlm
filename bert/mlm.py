

import torch.nn as nn

from .transformer import Transformer


class Generator(nn.Module):
    """
    Mask generator for guided mask language modeling
    Very inefficient first attempt
    """

    def __init__(self,
                 vocab_size,
                 maxlen=512,
                 n_layers=4,
                 d_model=128,
                 n_head=8,
                 p=0.1
                 ):

        super(Generator, self).__init__()

        self.transformer = Transformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        self.linear = nn.Linear(d_model, maxlen)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids):
        """
        :param ids:  [batch_size, maxlen]
        :return:     [batch_size, maxlen]
        """

        x = self.transformer(ids)
        x = x.mean(dim=1)
        x = self.sigmoid(self.linear(x))

        return x


class Discriminator(nn.Module):
    """
    Discriminator to predict BERT std given ids and masks
    for guided mask language modeling
    """

    def __init__(self,
                 vocab_size,
                 n_layers=4,
                 d_model=128,
                 n_head=8,
                 p=0.1
                 ):
        super(Discriminator, self).__init__()

        self.transformer = Transformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        self.linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 1)
        self.gelu = nn.GELU()

    def forward(self, ids, mask_score):
        """
        :param ids:        [batch_size, maxlen]
        :param mask_score: [batch_size, maxlen]
        :return:           [batch_size, 1]
        """

        x = self.transformer(ids, mask_score)
        x = x.mean(dim=1)
        x = self.gelu(self.linear(x))
        x = self.out(x)

        return x

