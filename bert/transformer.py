

import torch
import torch.nn as nn


from .embedding import TransformerEmbedding
from .layers import AttentionLayer


class Transformer(nn.Module):
    """
    Standard Transformer
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=12,
                 d_model=768,
                 n_head=8,
                 p=0.1
                 ):

        super(Transformer, self).__init__()

        self.emb = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len
        )
        self.layers = nn.ModuleList([
            AttentionLayer(
                d_model=d_model,
                ffn_hidden=4 * d_model,
                n_head=n_head,
                p=p
            )
            for _ in range(n_layers)
        ])

    def forward(self, ids):
        """
        :param   [batch_size, length]
        :return: [batch_size, d_model]
        """
        x = self.emb(ids)

        for layer in self.layers:
            x = layer(x)

        return x
