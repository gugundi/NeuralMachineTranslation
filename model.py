import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        # self.encoder = ...
        # self.decoder = ...
        # self.attention = ...
        self.tmp = nn.Linear(10, 10)
        self.out = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.tmp(x))
        x = self.out(x)
        return x
