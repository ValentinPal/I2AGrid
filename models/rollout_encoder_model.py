import numpy as np
import torch
from torch import nn as nn

class RolloutEncoder(nn.Module):
    def __init__(self, input_shape, hidden_size, conv_layers):
        super(RolloutEncoder, self).__init__()

        modules = []
        modules.append(nn.Conv2d(input_shape[0], conv_layers[0]['out_c'], conv_layers[0]['k'], conv_layers[0]['s']))
        modules.append(nn.ReLU())
        for i in range(1, len(conv_layers)):
            modules.append(
                nn.Conv2d(conv_layers[i]['in_c'], conv_layers[i]['out_c'], conv_layers[i]['k'], conv_layers[i]['s']))
            modules.append(nn.ReLU())

        self.conv = nn.Sequential(*modules)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1),
        #     nn.ReLU(),
        # )

        conv_out_size = self._get_conv_out(input_shape)

        self.rnn = nn.LSTM(input_size=conv_out_size+1, hidden_size=hidden_size, batch_first=False)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs_v, reward_v):
        """
        Input is in (time, batch, *) order
        """
        n_time = obs_v.size()[0]
        n_batch = obs_v.size()[1]
        n_items = n_time * n_batch
        obs_flat_v = obs_v.view(n_items, *obs_v.size()[2:])
        conv_out = self.conv(obs_flat_v)
        conv_out = conv_out.view(n_time, n_batch, -1)
        rnn_in = torch.cat((conv_out, reward_v), dim=2)
        _, (rnn_hid, _) = self.rnn(rnn_in)
        return rnn_hid.view(-1)
