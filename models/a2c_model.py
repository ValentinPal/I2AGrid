import numpy as np
import torch
from torch import nn as nn


class A2CModel(nn.Module):

    def __init__(self, input_shape, n_actions, conv_layers, policy_layer, value_layer, fc_layer, grad_cam = False):
        super(A2CModel, self).__init__()

        self.grad_cam = grad_cam
        self.last_conv_layer_grad = None

        modules = []
        modules.append(nn.Conv2d(input_shape[0], conv_layers[0]['out_c'], conv_layers[0]['k'], conv_layers[0]['s']))
        modules.append(nn.ReLU())
        for i in range(1, len(conv_layers)):
            modules.append(
                nn.Conv2d(conv_layers[i]['in_c'], conv_layers[i]['out_c'], conv_layers[i]['k'], conv_layers[i]['s']))
            modules.append(nn.ReLU())

        self.conv = nn.Sequential(*modules)

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
        conv_out = self.conv(fx)#.view(fx.size()[0], -1)
        if self.grad_cam:
            conv_out.register_hook(self.activations_hook)
        fc_out = self.fc(conv_out.view(fx.size()[0], -1))
        return self.policy(fc_out), self.value(fc_out)

    def activations_hook(self, grad):
        self.last_conv_layer_grad = grad

    def get_activations_gradient(self):
        return self.last_conv_layer_grad

    def get_activations(self, x):
        return self.conv(x.float())
