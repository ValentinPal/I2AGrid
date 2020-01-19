import numpy as np
import torch
from torch import nn as nn


class AtariA2CSmall(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2CSmall, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            # nn.Linear(conv_out_size, 256),
            # nn.ReLU(),
            nn.Linear(conv_out_size, n_actions)
        )

        self.value = nn.Sequential(
            # nn.Linear(conv_out_size, 256),
            # nn.ReLU(),
            nn.Linear(conv_out_size, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float()# / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)