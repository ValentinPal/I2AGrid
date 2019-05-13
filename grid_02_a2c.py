#!/usr/bin/env python3
import os
import gym
import ptan
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim

import gym
import gym_RandomGoalsGrid

import random

from lib import common, i2a

NUM_ENVS = 32
SAVE_EVERY_BATCH = 1000
OBS_WEIGHT = 10.0
REWARD_WEIGHT = 1.0
BATCH_SIZE = 64

GAMMA = 0.99
VALUE_LOSS_COEF = 0.5

LEARNING_RATE = 0.0005


def iterate_batches(envs, net, device="cpu"):
    act_selector = ptan.actions.ProbabilityActionSelector()
    mb_obs = np.zeros((BATCH_SIZE, ) + common.IMG_SHAPE, dtype=np.uint8)
#    mb_obs_next = np.zeros((BATCH_SIZE, ) + i2a.EM_OUT_SHAPE, dtype=np.float32)
    mb_obs_next = np.zeros((BATCH_SIZE, ) + common.IMG_SHAPE, dtype=np.float32)
    mb_actions = np.zeros((BATCH_SIZE, ), dtype=np.int32)
    mb_rewards = np.zeros((BATCH_SIZE, ), dtype=np.float32)
    obs = [e.reset() for e in envs]
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    batch_idx = 0
    done_rewards = []
    done_steps = []

    while True:
        obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
        
        logits_v, values_v = net(obs_v)
        probs_v = F.softmax(logits_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = act_selector(probs)

        for e_idx, e in enumerate(envs):
            o, r, done, _ = e.step(actions[e_idx])
            mb_obs[batch_idx] = obs[e_idx]
            #???????????????????????????????????????????????????????????????????
#            mb_obs_next[batch_idx] = get_obs_diff(obs[e_idx], o)#???????????????????????????????????????????????????????????
            mb_obs_next[batch_idx] = o #this is the observation after the step was taken
            #???????????????????????????????????????????????????????????????????
            mb_actions[batch_idx] = actions[e_idx]
            mb_rewards[batch_idx] = r

            total_reward[e_idx] += r
            total_steps[e_idx] += 1

            batch_idx = (batch_idx + 1) % BATCH_SIZE
            #yields only if batch completed
            if batch_idx == 0:
                yield mb_obs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps
                done_rewards.clear()
                done_steps.clear()
            if done:
                o = e.reset()
                done_rewards.append(total_reward[e_idx])
                done_steps.append(total_steps[e_idx])
                total_reward[e_idx] = 0.0
                total_steps[e_idx] = 0
            obs[e_idx] = o

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=False, help="Name of the run", default="_fresh")
#    parser.add_argument("-m", "--model", required=False, help="File with model to load", default="runs/May06_01-24-05_valy_grid_01_a2c_True__21_True_RandomGoalsGrid3CFast-v0_64_1_32_0.0005_0.99_0.015_5_False_True_1_2803944495_True_1_-1_524231_16_32/best_RandomGoalsGrid3CFast-v0_0028.000_13000.dat")
#    parser.add_argument("-m", "--model", required=False, help="File with model to load", default="runs/May08_03-02-50_valy_grid_01_a2c_RandomGoalsGrid3CFast-v0_21_5_True_128_3_32_0.0008_0.99_0.015_1_3665658711/best_RandomGoalsGrid3CFast-v0_0026.600_6000.dat")
    parser.add_argument("-m", "--model", required=False, help="File with model to load", default="runs-9/May08_01-50-49_valy_grid_01_a2c_RandomGoalsGrid3CFast-v0_33_9_True_128_3_32_0.0005_0.99_0.015_1_1960534414/best_RandomGoalsGrid3CFast-v0_0014.800_10000.dat")
#    parser.add_argument("-m", "--model", required=False, help="File with model to load", default="runs/May06_01-23-59_valy_grid_01_a2c_True__21_True_RandomGoalsGrid3CFast-v0_64_1_32_0.0005_0.99_0.015_5_False_True_1_3185480077_True_1_-1_524231_16_32/best_RandomGoalsGrid3CFast-v0_0027.000_20000.dat")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    SEED = random.randint(0, 2**32 - 1)
    
    NameForWriter = common.ENV_NAME + "_" + str(common.FRAME_SIZE) + "_" + str(common.GRID_SIZE) + "_" + str(common.CONV_LARGE) + "_" + \
                    str(BATCH_SIZE) + "_" + str(common.REWARD_STEPS) + "_" +  str(NUM_ENVS) + "_" + str(LEARNING_RATE) + "_" + \
                    str(GAMMA) + "_" + str(common.ENTROPY_BETA) + "_"  + str(common.CLIP_GRAD) + "_" + str(SEED) 
    writer = SummaryWriter(comment = "grid_02_a2c_" + NameForWriter)
    saves_path = writer.log_dir

    envs = [common.makeCustomizedGridEnv() for _ in range(NUM_ENVS)]
    
    net = common.getNet(device)

    #sets seed on torch operations and on all environments
    common.set_seed(SEED, envs=envs)

#    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n)
    net_em = i2a.EnvironmentModel(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
#    net_em.load_state_dict(torch.load("saves/02_env_PongNoFrameskip-v4__fresh/em_PongNoFrameskip-v4_46000_1.1654e-03.dat", map_location=lambda storage, loc: storage))
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    net = net.to(device)
    print(net_em)
    optimizer = optim.Adam(net_em.parameters(), lr=LEARNING_RATE)

    step_idx = 0
    best_loss = np.inf
    with ptan.common.utils.TBMeanTracker(writer, batch_size=BATCH_SIZE) as tb_tracker:
        #obtain batch transitions from the a2c model free agent (st, at, st+1, r)
        for mb_obs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps in iterate_batches(envs, net, device):
            if len(done_rewards) > 0:
                m_reward = np.mean(done_rewards)
                m_steps = np.mean(done_steps)
                print("%d: done %d episodes, mean reward=%.2f, steps=%.2f" % (
                    step_idx, len(done_rewards), m_reward, m_steps))
                tb_tracker.track("total_reward", m_reward, step_idx)
                tb_tracker.track("total_steps", m_steps, step_idx)

            #transform the arrays into tensors
            #TODO: see if tensors can be used directly in the iterator, to avoid the transfer to GPU
            obs_v = torch.FloatTensor(mb_obs).to(device)
            obs_next_v = torch.FloatTensor(mb_obs_next).to(device)
            actions_t = torch.LongTensor(mb_actions.tolist()).to(device)
            rewards_v = torch.FloatTensor(mb_rewards).to(device)

            optimizer.zero_grad()
            
            #obtain a predictions for next state (observation) and future reward, based on current obs from the above
            #batches and the actions that are coming from the A2C model free agent
            out_obs_next_v, out_reward_v = net_em(obs_v.float()/255, actions_t)
            loss_obs_v = F.mse_loss(out_obs_next_v.squeeze(-1), obs_next_v)
            loss_rew_v = F.mse_loss(out_reward_v.squeeze(-1), rewards_v)
            
            #loss combines the rewards loss and the observation prediction loss, each of them weighed by some coeff
            #TODO: use maybe a separate head and a common boddy network idea
            #since the gradients for reward and observation loss will have different "dynamics"
            loss_total_v = OBS_WEIGHT * loss_obs_v + REWARD_WEIGHT * loss_rew_v
            loss_total_v.backward()
            optimizer.step()
            tb_tracker.track("loss_em_obs", loss_obs_v, step_idx)
            tb_tracker.track("loss_em_reward", loss_rew_v, step_idx)
            tb_tracker.track("loss_em_total", loss_total_v, step_idx)

            loss = loss_total_v.data.cpu().numpy()
            if loss < best_loss:
                print("Best loss updated: %.4e -> %.4e" % (best_loss, loss))
                best_loss = loss
                fname = os.path.join(saves_path, "best_%s_%.4e_%05d.dat" % (common.ENV_NAME, loss, step_idx))
                torch.save(net_em.state_dict(), fname)

            step_idx += 1
            if step_idx % SAVE_EVERY_BATCH == 0:
                fname = os.path.join(saves_path, "em_%s_%05d_%.4e.dat" % (common.ENV_NAME, step_idx, loss))
                torch.save(net_em.state_dict(), fname)
# -*- coding: utf-8 -*-

