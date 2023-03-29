
import torch
import torch.nn as nn
from torch.optim import Adam


import tqdm


class BERTTrainer:

    def __init__(self,
                 bert,
                 generator,
                 discriminator,
                 train_dataloader,
                 test_dataloader,
                 batch_size=32,
                 lr=1e-4
                 ):

        self.train_data = train_dataloader
        self.test_data = test_dataloader

    def train(self, epoch):
        self.train_iteration(epoch, self.train_data)

    def test(self, epoch):
        self.test_iteration(epoch, self.test_data)

    def train_iteration(self, epoch, data_loader):
        pass

    def test_iteration(self, epoch, data_loader):
        pass

    def eval(self, bert_input, bert_target, device="cuda"):
        pass

    def update(self, bert_input, bert_target, device="cuda"):
        pass

    def bert_loss(self, target, expected):
        return self.criterion(expected, target)

    def save(self, file_path="saved/trained"):
        bert_path = file_path + ".bert"
        torch.save(self.bert, bert_path)
