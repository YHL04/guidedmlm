

import torch
import torch.nn as nn
import numpy as np

from .transformer import Transformer, DiscriminatorTransformer


def gumbel(shape, eps=1e-10):
    """ Sample from Gumbel(0, 1)"""
    u = np.random.random(shape)
    g = -np.log(-np.log(u + eps) + eps)
    return g


def gumbel_max_sample(x, is_prob=False):
    """ Draw a sample from P(X=k) prop x_k """
    if is_prob:
        x = np.log(x)

    g = gumbel(shape=x.shape)
    return (g + x).argmax(axis=0)


def softmax(X, temperature=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    temperature (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y / float(temperature)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)

    # take the sum along the specified axis
    p = y / np.expand_dims(np.sum(y, axis = axis), axis)

    if len(X.shape) == 1:
        p = p.flatten()

    return p


def gumbel_softmax_sample(logits, temperature=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + gumbel(logits.shape)
    return softmax(y, temperature=temperature)


def gumbel_softmax(logits, temperature, hard=False):
    '''Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    '''
    y = gumbel_softmax_sample(logits, temperature)

    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y

    return y


class Generator(nn.Module):
    """
    Mask generator for guided mask language modeling
    Very inefficient first attempt
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=128,
                 n_head=8,
                 p=0.1
                 ):
        super(Generator, self).__init__()

        self.transformer = Transformer(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        self.linear = nn.Linear(d_model, max_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids):
        """
        :param ids:  [batch_size, maxlen]
        :return:     [batch_size, maxlen]
        """

        x = self.transformer(ids)
        x = self.sigmoid(self.linear(x.mean(dim=1)))

        return x


class Discriminator(nn.Module):
    """
    Discriminator to predict BERT std given ids and masks
    for guided mask language modeling
    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 n_layers,
                 d_model,
                 n_head,
                 p
                 ):
        super(Discriminator, self).__init__()

        self.transformer = DiscriminatorTransformer(
            vocab_size=vocab_size,
            max_len=max_len,
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
        x = self.gelu(self.linear(x.mean(dim=1)))
        x = self.out(x)

        return x

