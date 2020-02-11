import numpy as np
import torch
from torch import nn as nn


class A2CModel(nn.Module):

    def __init__(self, input_shape, n_actions, conv_layers, policy_layer, value_layer, fc_layer, config):
        super(A2CModel, self).__init__()

        modules = []
        modules.append(nn.Conv2d(input_shape[0], conv_layers[0]['out_c'], conv_layers[0]['k'], conv_layers[0]['s']))
        modules.append(nn.ReLU())
        for i in range(1, len(conv_layers)):
            modules.append(
                nn.Conv2d(conv_layers[i]['in_c'], conv_layers[i]['out_c'], conv_layers[i]['k'], conv_layers[i]['s']))
            modules.append(nn.ReLU())

        self.conv = nn.Sequential(*modules)

        self.deconv = nn.ConvTranspose2d(config.EM_DECONV['in_c'], config.EM_DECONV['out_c'],
                                         kernel_size=config.EM_DECONV['k'], stride=config.EM_DECONV['s'],
                                         padding=config.EM_DECONV['p'])

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, fc_layer),
            nn.ReLU()
        )
        self.policy = nn.Linear(policy_layer, n_actions)
        self.value = nn.Linear(value_layer, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, x):
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        fc_out = self.fc(conv_out)
        return self.policy(fc_out), self.value(fc_out)