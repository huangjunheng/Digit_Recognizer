import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self):

        super(MLP, self).__init__()

        self.func = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 10),
        )


    def forward(self, x):

        x = self.func(x)
        return x
