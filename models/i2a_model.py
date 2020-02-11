import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rollout_encoder_model import RolloutEncoder


class I2A(nn.Module):
    def __init__(self, input_shape, n_actions, net_em, net_policy, config, grad_cam = False):
        super(I2A, self).__init__()

        self.grad_cam = grad_cam
        self.last_conv_layer_grad = None

        self.n_actions = n_actions
        self.rollout_steps = config.ROLL_STEPS
        conv_layers = config.A2C_CONV_LAYERS
        modules = []
        modules.append(nn.Conv2d(input_shape[0], conv_layers[0]['out_c'], conv_layers[0]['k'], conv_layers[0]['s']))
        modules.append(nn.ReLU())
        for i in range(1, len(conv_layers)):
            modules.append(
                nn.Conv2d(conv_layers[i]['in_c'], conv_layers[i]['out_c'], conv_layers[i]['k'], conv_layers[i]['s']))
            modules.append(nn.ReLU())

        self.conv = nn.Sequential(*modules)
        conv_out_size = self._get_conv_out(input_shape)
        #rollouts are always performed for all actions available
        #so the input size for policy and V(s) need to account for all rollouts of all actions
        fc_input = conv_out_size + config.ROLLOUT_HIDDEN * n_actions

        self.fc = nn.Sequential(
            nn.Linear(fc_input, config.FC_LAYER),
            nn.ReLU()
        )
        self.policy = nn.Linear(config.POLICY_LAYER, n_actions)
        self.value = nn.Linear(config.VALUE_LAYER, 1)

        self.encoder = RolloutEncoder(config.IMG_SHAPE, config.ROLLOUT_HIDDEN, config.A2C_CONV_LAYERS)
        self.action_selector = ptan.actions.ProbabilityActionSelector()
        # save refs without registering
        object.__setattr__(self, "net_em", net_em)
        object.__setattr__(self, "net_policy", net_policy)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float()
        enc_rollouts = self.rollouts_batch(fx)
        conv_out = self.conv(fx)
        if self.grad_cam:
            conv_out.register_hook(self.activations_hook)
        #input to the fc layer requires all the rollouts concatenated
        fc_in = torch.cat((conv_out.view(fx.size()[0], -1), enc_rollouts), dim=1)
        fc_out = self.fc(fc_in)
        return self.policy(fc_out), self.value(fc_out)

    def rollouts_batch(self, batch):
        batch_size = batch.size()[0]
        batch_rest = batch.size()[1:]
        if batch_size == 1:
            obs_batch_v = batch.expand(batch_size * self.n_actions, *batch_rest)
        else:
            obs_batch_v = batch.unsqueeze(1)
            obs_batch_v = obs_batch_v.expand(batch_size, self.n_actions, *batch_rest)
            obs_batch_v = obs_batch_v.contiguous().view(-1, *batch_rest)
        actions = np.tile(np.arange(0, self.n_actions, dtype=np.int64), batch_size)
        step_obs, step_rewards = [], []

        for step_idx in range(self.rollout_steps):
            actions_t = torch.tensor(actions, dtype=torch.int64).to(batch.device)
            obs_next_v, reward_v = self.net_em(obs_batch_v, actions_t)
            step_obs.append(obs_next_v.detach())
            step_rewards.append(reward_v.detach())
            # don't need actions for the last step
            if step_idx == self.rollout_steps-1:
                break

            logits_v, _ = self.net_policy(obs_batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            actions = self.action_selector(probs)
        step_obs_v = torch.stack(step_obs)
        step_rewards_v = torch.stack(step_rewards)
        flat_enc_v = self.encoder(step_obs_v, step_rewards_v)
        return flat_enc_v.view(batch_size, -1)

    def activations_hook(self, grad):
        self.last_conv_layer_grad = grad

    def get_activations_gradient(self):
        return self.last_conv_layer_grad

    def get_activations(self, x):
        return self.conv(x.float())

