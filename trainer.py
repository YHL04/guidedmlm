

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import random
import datetime


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
        self.criterion = nn.NLLLoss(ignore_index=0, reduction="none")

        self.b_optimizer = Adam(self.bert.parameters(), lr=lr)
        self.g_optimizer = Adam(self.generator.parameters(), lr=lr)
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=lr)

        self.datetime = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
        self.file = open(f"logs/{self.datetime}", "w")

    @staticmethod
    def mask_ids(tokens, p):
        """
        :param tokens: Tensor[total_len, max_len]
        :param p:      mask probability
        """

        target = torch.zeros_like(tokens, dtype=torch.int64).cuda()

        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                prob = random.random()

                if prob < p:
                    # index for [MASK] is 103
                    target[i][j] = tokens[i][j]
                    tokens[i][j] = 103

        return tokens, target

    def train_step(self):
        batch = self.memory.get_batch(batch_size=self.batch_size)
        tokens, target = self.mask_ids(batch, p=0.20)

        expected = self.bert.forward(tokens)
        loss = self.bert_loss(expected, target)
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.bert.parameters(), 0.1)
        self.b_optimizer.step()
        loss = loss.item()

        self.file.write("{}\n".format(loss))
        self.file.flush()

        return loss, 0.

    def apply_mask(self, batch, mask_prob, p=0.1):
        assert batch.shape == (self.batch_size, 512)
        assert mask_prob.shape == (self.batch_size, 512)

        target = torch.zeros(self.batch_size, 512).to(torch.int64).cuda()
        mask_idx = torch.topk(mask_prob, 512 // 10, dim=1).indices

        for i in range(len(mask_idx)):
            for j in range(len(mask_idx[i])):
                batchidx = i
                lengthidx = mask_idx[i][j]

                target[batchidx, lengthidx] = batch[batchidx, lengthidx]
                batch[batchidx, lengthidx] = 103

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                prob = random.random()

                if prob < p:
                    target[i][j] = batch[i][j]
                    batch[i][j] = 103

        return batch, target

    def train_step_2(self):
        batch = self.memory.get_batch(batch_size=self.batch_size)

        # get mask from generator
        with torch.no_grad():
            mask_prob = self.generator.forward(batch)
            tokens, target = self.apply_mask(batch, mask_prob)

        expected = self.bert.forward(tokens)
        loss = self.bert_loss(expected, target).mean(dim=1)

        target = loss.detach().view(self.batch_size, 1)
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.bert.parameters(), 0.1)
        self.b_optimizer.step()
        loss = loss.item()

        # train discriminator
        expected = self.discriminator.forward(batch, torch.rand(self.batch_size, 512).cuda())

        assert expected.shape == (self.batch_size, 1)
        assert target.shape == (self.batch_size, 1)

        d_loss = F.mse_loss(expected, target)
        d_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.1)
        self.d_optimizer.step()
        d_loss = d_loss.item()

        # train generator (generate mask that maximizes loss according to discriminator)
        mask_prob = self.generator.forward(batch)
        pred_loss = self.discriminator.forward(batch, mask_prob)
        pred_loss = -pred_loss.mean()
        pred_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.1)
        self.g_optimizer.step()

        self.file.write("{}, {}\n".format(loss, d_loss))
        self.file.flush()

        return loss, d_loss

    def bert_loss(self, expected, target):
        """
        :param expected: Tensor[batch_size, length, vocab_size]
        :param target  : Tensor[batch_size, length]
        """
        assert expected.shape == (self.batch_size, 512, 30522)
        assert target.shape == (self.batch_size, 512)

        expected = expected.transpose(1, 2)
        return self.criterion(expected, target)

    def save(self, file_path="saved/trained"):
        bert_path = file_path + ".bert"
        torch.save(self.bert, bert_path)

