#!/usr/bin/env python3
import os
import ptan
import time
import gym
import argparse
import random

import gym
import gym_RandomGoalsGrid

from lib import common, i2a
import sys

import torch.optim as optim
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter


common.REWARD_STEPS = 3
SAVE_EVERY_BATCH = 300
OBS_WEIGHT = 10.0
REWARD_WEIGHT = 1.0
BATCH_SIZE = 64

VALUE_LOSS_COEF = 0.5
GAMMA = 0.99

ENTROPY_BETA = 0.015

ROLLOUTS_STEPS = 5
LEARNING_RATE = 1e-3
POLICY_LR = 1e-4
TEST_EVERY_BATCH = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=False, help="Name of the run", default="new")
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable CUDA")
    parser.add_argument("--em", required=False, help="Environment model file name", default="runs/May08_03-20-01_valygrid_02_a2c_RandomGoalsGrid3CFast-v0_21_5_True_64_3_32_0.0005_0.99_0.015_1_3254759783/best_RandomGoalsGrid3CFast-v0_2.0368e-02_1144661.dat")
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED, help="Random seed to use, default=%d" % common.DEFAULT_SEED)
    parser.add_argument("--roll_steps", type=int, default=5, help="how many steps in the imagined trajectory")
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.cuda else "cpu")
    
    SEED = random.randint(0, 2**32 - 1)
    
    ROLLOUTS_STEPS = args.roll_steps
    
    NameForWriter = common.ENV_NAME + "_" + str(common.FRAME_SIZE) + "_" + str(BATCH_SIZE) + "_" + str(common.REWARD_STEPS) + "_" +  str(common.NUM_ENVS) + "_" + str(LEARNING_RATE) + "_" + \
                    str(GAMMA) + "_" + str(common.ENTROPY_BETA) + "_"  + str(common.GRID_SIZE) + "_" + \
                    str(SEED) + "_" + str(ROLLOUTS_STEPS)
    writer = SummaryWriter(comment = "grid_03_a2c_" + NameForWriter)
    saves_path = writer.log_dir

    envs = [common.makeCustomizedGridEnv() for _ in range(common.NUM_ENVS)]
    test_env = common.makeCustomizedGridEnv()
    
    #sets seed on torch operations and on all environments
    common.set_seed(SEED, envs=envs)
    common.set_seed(SEED, envs=[test_env])

    obs_shape = envs[0].observation_space.shape
    act_n = envs[0].action_space.n

#    net_policy = common.AtariA2C(obs_shape, act_n).to(device)
    net_policy = common.getNet(device)
    
    net_em = i2a.EnvironmentModel(obs_shape, act_n)
    net_em.load_state_dict(torch.load(args.em, map_location=lambda storage, loc: storage))
    net_em = net_em.to(device)

    net_i2a = i2a.I2A(obs_shape, act_n, net_em, net_policy, ROLLOUTS_STEPS).to(device)
#    net_i2a.load_state_dict(torch.load("saves/03_i2a_test/best_pong_-018.667_1300.dat", map_location=lambda storage, loc: storage))
    print(net_i2a)
    print("em param count: ", common.count_parameters(net_em))
    print("net_policy param count: ", common.count_parameters(net_policy))
    print("ia policy param count: ", common.count_parameters(net_i2a))

    obs = envs[0].reset()
    obs_v = ptan.agent.default_states_preprocessor([obs]).to(device)
    res = net_i2a(obs_v)

    optimizer = optim.RMSprop(net_i2a.parameters(), lr=LEARNING_RATE, eps=1e-5)
    policy_opt = optim.Adam(net_policy.parameters(), lr=POLICY_LR)

    step_idx = 0
    total_steps = 0
    ts_start = time.time()
    best_reward = None
    best_test_reward = None
    with ptan.common.utils.TBMeanTracker(writer, batch_size=32) as tb_tracker:
        for mb_obs, mb_rewards, mb_actions, mb_values, mb_probs, done_rewards, done_steps in \
                common.iterate_batches(envs, net_i2a, device):
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
                mean_rw = done_rewards.mean()
                print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f, speed=%.2f f/s" % (
                    step_idx, len(done_rewards), mean_rw, best_reward, speed))
                

            obs_v = common.train_a2c(net_i2a, mb_obs, mb_rewards, mb_actions, mb_values,
                                     optimizer, tb_tracker, step_idx, device=device)
            # policy distillation
            probs_v = torch.FloatTensor(mb_probs).to(device)
            policy_opt.zero_grad()
            logits_v, _ = net_policy(obs_v)
            policy_loss_v = -F.log_softmax(logits_v, dim=1) * probs_v.view_as(logits_v)
            policy_loss_v = policy_loss_v.sum(dim=1).mean()
            policy_loss_v.backward()
            policy_opt.step()
            tb_tracker.track("loss_distill", policy_loss_v, step_idx)

            step_idx += 1

            if step_idx % TEST_EVERY_BATCH == 0:
                test_reward, test_steps = common.test_model(test_env, net_i2a, device=device)
                writer.add_scalar("test_reward", test_reward, step_idx)
                writer.add_scalar("test_steps", test_steps, step_idx)
                if best_test_reward is None or best_test_reward < test_reward:
                    if best_test_reward is not None:
                        fname = os.path.join(saves_path, "best_%s_%08.3f_%d.dat" % (common.ENV_NAME, test_reward, step_idx))
                        torch.save(net_i2a.state_dict(), fname)
                        torch.save(net_policy.state_dict(), fname + ".policy")
                    else:
                        fname = os.path.join(saves_path, "em.dat")
                        torch.save(net_em.state_dict(), fname)
                    best_test_reward = test_reward
                print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                    step_idx, test_reward, test_steps, best_test_reward))
            
            if step_idx > 10000:
                break
# -*- coding: utf-8 -*-

