

import torch
import torch.nn as nn
from torch.optim import Adam

import random


class Trainer:

    def __init__(self,
                 bert,
                 generator,
                 discriminator,
                 memory,
                 lr=1e-4,
                 batch_size=32,
                 tau=0.01
                 ):

        self.bert = bert
        self.generator = generator
        self.discriminator = discriminator

        self.batch_size = batch_size
        self.tau = tau

        self.memory = memory
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.optimizer = Adam(self.bert.parameters(), lr=lr)

    def mask_ids(self, tokens, p):
        """
        :param tokens: Tensor[total_len, max_len]
        :param p:      mask probability
        """

        target = torch.zeros(*tokens.size(), dtype=torch.int64)

        for i in range(len(tokens)):
            for j in range(len(tokens)):
                prob = random.random()

                if prob < p:
                    # index for [MASK] is 103
                    target[i][j] = tokens[i][j]
                    tokens[i][j] = 103

            return tokens, target

    def train_step(self):
        batch = self.memory.get_batch(batch_size=self.batch_size)
        tokens, target = self.mask_ids(batch)

        expected = self.bert.forward(tokens)
        loss = self.bert_loss(expected, target)
        loss.backward()

        self.optimizer.step()

        return loss

    def bert_loss(self, expected, target):
        return self.criterion(expected, target)

    def save(self, file_path="saved/trained"):
        bert_path = file_path + ".bert"
        torch.save(self.bert, bert_path)

