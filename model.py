import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.tmp = nn.Linear(10, 10)
        # self.encoder = ...
        # self.decoder = ...
        # self.attention = ...
        pass

    def forward(self, x):
        return x
