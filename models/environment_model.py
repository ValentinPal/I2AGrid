import numpy as np
import torch
from torch import nn as nn


class EnvironmentModel(nn.Module):
    def __init__(self, input_shape, n_actions, config):
        super(EnvironmentModel, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions
        conv_layers = config.EM_CONV1
        # input color planes will be equal to frames plus one-hot encoded actions
        n_planes = input_shape[0] + n_actions
        modules = []
        modules.append(nn.Conv2d(n_planes, conv_layers[0]['out_c'], conv_layers[0]['k'], conv_layers[0]['s']))
        modules.append(nn.ReLU())
        for i in range(1, len(conv_layers)):
            modules.append(nn.Conv2d(conv_layers[i]['in_c'], conv_layers[i]['out_c'], conv_layers[i]['k'], conv_layers[i]['s'], conv_layers[i]['p']))
            modules.append(nn.ReLU())
        self.conv1 = nn.Sequential(*modules)

        self.conv2 = nn.Sequential(
            nn.Conv2d(config.EM_CONV2['in_c'], config.EM_CONV2['out_c'], kernel_size= config.EM_CONV2['k'], padding = config.EM_CONV2['p']),
            nn.ReLU()
        )
        self.deconv = nn.ConvTranspose2d(config.EM_DECONV['in_c'], config.EM_DECONV['out_c'], kernel_size= config.EM_DECONV['k'], stride = config.EM_DECONV['s'], padding=config.EM_DECONV['p'])

        self.reward_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        rw_conv_out = self._get_reward_conv_out((n_planes, ) + input_shape[1:])
        self.reward_fc = nn.Sequential(
            nn.Linear(rw_conv_out, config.EM_RW_FC),
            nn.ReLU(),
            nn.Linear(config.EM_RW_FC, 1)
        )

    def _get_reward_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.reward_conv(o)
        return int(np.prod(o.size()))

    def forward(self, imgs, actions):
        batch_size = actions.size()[0]

        #creates a zeroed out plane/matrix for each action
        act_planes_v = torch.FloatTensor(batch_size, self.n_actions, *self.input_shape[1:]).zero_().to(actions.device)

        #fills the plane(s) corresponding to the action(s) with 1. (s) is because we work in batches
        act_planes_v[range(batch_size), actions] = 1.0

        comb_input_v = torch.cat((imgs, act_planes_v), dim=1)
        c1_out = self.conv1(comb_input_v)
        c2_out = self.conv2(c1_out)
        c2_out += c1_out
        img_out = self.deconv(c2_out)
        rew_conv = self.reward_conv(c2_out).view(batch_size, -1)
        rew_out = self.reward_fc(rew_conv)
        return img_out, rew_out