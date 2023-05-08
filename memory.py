

import torch
import random

random.seed(0)


class Memory:

    def __init__(self,
                 data,
                 dim=512
                 ):
        """
        :param data:   List[Tensor(length, d_model)]
        """
        self.data = data
        self.size = len(data)

        self.dim = dim

    def get_batch(self, batch_size=32):
        x = []

        for i in range(batch_size):
            buffer_idx = random.randrange(0, self.size)
            time_idx = random.randrange(0, self.data[buffer_idx].size(0))

            tokens = self.data[buffer_idx][time_idx]
            x.append(tokens)

        x = torch.stack(x).cuda()
        assert x.shape == (batch_size, x.size(1))

        return x

