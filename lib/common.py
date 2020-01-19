import numpy as np
import gym
import torch
from models.a2cmodel import A2CModel
import gym_RandomGoalsGrid

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def getNet(device, cfg):
    net = None
    fs = cfg.FRAME_SIZE
    net = A2CModel(cfg.IMG_SHAPE, 4, cfg.A2C_CONV_LAYERS, cfg.POLICY_LAYER, cfg.VALUE_LAYER, cfg.FC_LAYER).to(device)
    print(net)
    return net

def makeCustomizedGridEnv(cfg):
    env = gym.make(cfg.ENV_NAME)
    env.customInit(size = cfg.GRID_SIZE,
            partial = cfg.PARTIALLY_OBSERVED_GRID,
            pixelsEnv = cfg.PIXELS_ENV,
            frameSize = cfg.FRAME_SIZE,
            replacement = cfg.REPLACEMENT,
            negativeReward = cfg.NEGATIVE_RW,
            positiveReward = cfg.POSITIVE_RW)

    return env

def set_seed(seed, envs=None, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    if envs:
        for idx, env in enumerate(envs):
            # required to have different seed per environment, so not to gather the same tuples from all environments
            env.seed(seed + idx)

def convert_conv_cfg_to_str(config):
    config.A2C_CONV_LAYERS = str([layer[param] for layer in config.A2C_CONV_LAYERS for param in layer])
    config.EM_CONV1 = str([layer[param] for layer in config.EM_CONV1 for param in layer])
    config.EM_CONV2 = str([config.EM_CONV2[param] for param in config.EM_CONV2])
    config.EM_DECONV = str([config.EM_DECONV[param] for param in config.EM_DECONV])
    config.IMG_SHAPE = str(config.IMG_SHAPE)

def save_exp_config_to_TB(writer, config):
    convert_conv_cfg_to_str(config)
    # metric_dict = {'Z_B_RW': best_reward, 'Z_B_T_RW': best_test_reward}
    writer.add_hparams(hparam_dict=vars(config), metric_dict={})


def save_EM_exp_config_to_TB(writer, config):
    convert_conv_cfg_to_str(config)
    # metric_dict = {'Z_B_LOSS': best_loss}
    writer.add_hparams(hparam_dict=vars(config), metric_dict={})