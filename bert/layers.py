

import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention for Transformers

    Query : [batch_size, head, length, d_tensor]
    Key : T [batch_size, head, d_tensor, length]
    Value : [batch_size, head, length, d_tensor]

    score : [batch_size, head, length, length]
    v_out : [batch_size, head, length, d_tensor]
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.shape

        k_t = k.transpose(2, 3)
        score = torch.matmul(q, k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = self.softmax(score)

        v = torch.matmul(score, v)

        return v, score


class Attention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        :param q, k, v: [batch_size, length, d_model]
        :return: out:   [batch_size, length, d_model]
        """
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        Split tensor into number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.shape

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.shape
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class AttentionLayer(nn.Module):
    """
    Standard transformer layer
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(AttentionLayer, self).__init__()
        self.attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class FeedForward(nn.Module):
    """
    Sequential(
        Linear(dim, inner_dim)
        GELU()
        Linear(inner_dim, dim)
    )
    """

    def __init__(self, dim, inner_dim):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)

