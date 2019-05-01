#!/usr/bin/env python3
import os
import ptan
import numpy as np
import time
import argparse
import gym
from tensorboardX import SummaryWriter
import torch.nn.utils as nn_utils
import torch
import torch.optim as optim

import gym
import gym_RandomGoalsGrid
import ptan

import random
import sys

from lib import common

import torch.nn as nn
import torch.nn.functional as F

NUM_ENVS = 32


REWARD_STEPS = 1

VALUE_LOSS_COEF = 0.5

GAMMA = 0.99
LEARNING_RATE = 0.0005
ENTROPY_BETA = 0.015
BATCH_SIZE = 32


TEST_EVERY_BATCH = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=False, help="Name of the run", default="_fresh")
    parser.add_argument("--seed", type=int, default=20, help="Random seed to use, default=%d" % 20)
    parser.add_argument("--steps", type=int, default=None, help="Limit of training steps, default=disabled")
    parser.add_argument("--gridSize", type=int, default=5, help="The size of the grid, default=5", required = False)
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.cuda else "cpu")
    
    SEED = random.randint(0, 2**32 - 1)
    
    common.GRID_SIZE = args.gridSize
    
    #NameForWriter includes all the params/hyperparams used in the first step of the I2A (baseline agent)
    #the folder that will be named with this string will contains both the logs for tensorboard and the saved net params.
    NameForWriter = str(common.pixelsEnv) + "_" + "_" + str(common.FRAME_SIZE) + "_" + str(common.CONV_LARGE) + "_" + common.ENV_NAME + "_" + \
                    str(BATCH_SIZE) + "_" + str(REWARD_STEPS) + "_" +  str(NUM_ENVS) + "_" + str(LEARNING_RATE) + "_" + \
                    str(GAMMA) + "_" + str(ENTROPY_BETA) + "_"  + str(common.GRID_SIZE) + "_" + \
                    str(common.PARTIALLY_OBSERVED_GRID) + "_" + str(common.REPLACEMENT) + "_" + str(CLIP_GRAD) + "_" + \
                    str(SEED) + "_" + str(common.USE_FRAMESTACK_WRAPPER) + "_" + str(common.POSITIVE_RW) + "_" + str(common.NEGATIVE_RW) + "_" + str(common.SMALL_CONV_NET_CFG)
    writer = SummaryWriter(comment = "_grid_01_a2c_" + NameForWriter)
    writer.log_dir = writer.log_dir.replace("runs", "runs-" + str(common.GRID_SIZE))
    saves_path = writer.log_dir

    #envs used for sampling tuples of experience    
    envs = [common.makeCustomizedGridEnv() for _ in range(NUM_ENVS)]
    #env used to test the avg reward produced by the current best net
    test_env = common.makeCustomizedGridEnv()

    net = common.getNet(device)
    print(common.count_parameters(net))
    
    #sets seed on torch operations and on all environments
    common.set_seed(SEED, envs=envs)
    common.set_seed(SEED, envs=[test_env])
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    step_idx = 0
    total_steps = 0
    best_reward = None
    ts_start = time.time()
    best_test_reward = None
    with ptan.common.utils.TBMeanTracker(writer, batch_size=BATCH_SIZE) as tb_tracker:
        for mb_obs, mb_rewards, mb_actions, mb_values, _, done_rewards, done_steps in \
                common.iterate_batches(envs, net, device=device):
            if len(done_rewards) > 0:
                total_steps += sum(done_steps)
                speed = total_steps / (time.time() - ts_start)
                if best_reward is None:
                    best_reward = done_rewards.max()
                elif best_reward < done_rewards.max():
                    best_reward = done_rewards.max()
                tb_tracker.track("total_reward_max", best_reward, step_idx)
                tb_tracker.track("total_reward", done_rewards, step_idx)
                tb_tracker.track("total_steps", done_steps, step_idx)
                print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f, speed=%.2f" % (
                    step_idx, len(done_rewards), done_rewards.mean(), best_reward, speed))

            common.train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values,
                             optimizer, tb_tracker, step_idx, device=device)
            step_idx += 1
            if args.steps is not None and args.steps < step_idx:
                break

            if step_idx % TEST_EVERY_BATCH == 0:
                test_reward, test_steps = common.test_model(test_env, net, device=device)
                writer.add_scalar("test_reward", test_reward, step_idx)
                writer.add_scalar("test_steps", test_steps, step_idx)
                if best_test_reward is None or best_test_reward < test_reward:
                    if best_test_reward is not None:
                        fname = os.path.join(saves_path, "best_%s_%08.3f_%d.dat" % (common.ENV_NAME,test_reward, step_idx))
                        torch.save(net.state_dict(), fname)
                    best_test_reward = test_reward
                print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                    step_idx, test_reward, test_steps, best_test_reward))
