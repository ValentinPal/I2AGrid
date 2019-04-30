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

from lib import common_new
from lib import common

import torch.nn as nn
import torch.nn.functional as F

NUM_ENVS = 32
REWARD_STEPS = 1

ENTROPY_BETA = 0.02
VALUE_LOSS_COEF = 0.5

GAMMA = 0.99
LEARNING_RATE = 0.0005
ENTROPY_BETA = 0.015
BATCH_SIZE = 64
CLIP_GRAD = 1

channels = 3
#
#common.FRAMES_COUNT = 3
#common.IMG_SHAPE = (channels * common.FRAMES_COUNT, FRAME_SIZE, FRAME_SIZE)

TEST_EVERY_BATCH = 1000
SEED = 50

import random
import sys


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def iterate_batches(envs, net, device="cuda"):
    n_actions = 4
    act_selector = ptan.actions.ProbabilityActionSelector()
    obs = [e.reset() for e in envs]
    batch_dones = [[False] for _ in range(NUM_ENVS)]
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    mb_obs = np.zeros((NUM_ENVS, REWARD_STEPS) + common.IMG_SHAPE, dtype=np.float32)
    mb_rewards = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_values = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_actions = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.int32)
    mb_probs = np.zeros((NUM_ENVS, REWARD_STEPS, n_actions), dtype=np.float32)
    
    #one iteration computes partial trajectories (n step) for all environments
    #1. forward pass to get action logits and V(s) for a specified observation
    #2. select the action based on action selector (using action probabilities given by the model) and executes it
    #3. store rewards per step per env, total reward per env (per trajectory), "dones", last obs
    #4. compute total discounted reward for the n-step trajectory using V(s)*(gamma**n) at the end if trajectory did not reach "done"
    #5. yield returns last observations per trajectory(per env), actions taken, V(s), a list of total rewards for done episodes,
    # a list of the steps where episodes finished
    # next time the iterator calls this method, this function will do another while iteration
    while True:
        batch_dones = [[dones[-1]] for dones in batch_dones]
        done_rewards = []
        done_steps = []
        for n in range(REWARD_STEPS):#interact with all envs for every step of a partial trajectory
            obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
            mb_obs[:, n] = obs_v.data.cpu().numpy()
            logits_v, values_v = net(obs_v)#forward pass, to get the logits for the actions and the V of the state/obs
            probs_v = F.softmax(logits_v, dim=1)#probabilities of the logits from the AC (it can also be i2a) model
            probs = probs_v.data.cpu().numpy()
            actions = act_selector(probs)#selects an action by sampling according to the probs given by AC model(or i2a)
            mb_probs[:, n] = probs
            mb_actions[:, n] = actions#one action per environment, for the current step of the trajectory
            mb_values[:, n] = values_v.squeeze().data.cpu().numpy()#value for every state from the trajectory
            for e_idx, e in enumerate(envs):
                o, r, done, _ = e.step(actions[e_idx])
                total_reward[e_idx] += r#computes partial trajectory rw per each env simulation
                total_steps[e_idx] += 1#trajectory steps per env
                if done:
                    o = e.reset()
                    done_rewards.append(total_reward[e_idx])
                    done_steps.append(total_steps[e_idx])
                    total_reward[e_idx] = 0.0
                    total_steps[e_idx] = 0
                obs[e_idx] = o
                mb_rewards[e_idx, n] = r#reward per env and per step for a trajectory
                batch_dones[e_idx].append(done)
        # obtain values for the last observation. obs holds last observation for every environment
        obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
        _, values_v = net(obs_v)
        values_last = values_v.squeeze().data.cpu().numpy()

        #compute the total discounted reward for an n-step trajectory, per environment
        #if last step is not end of trajectory, add to the rewards V(last_obs_of_traj)
        for e_idx, (rewards, dones, value) in enumerate(zip(mb_rewards, batch_dones, values_last)):
            rewards = rewards.tolist()#all rewards gotten for every step of this trajectory for this env
            #TODO: what if done is at the middle of a trajectory and then another one starts for the same env?
            #shouldn t we compute discounted reward up to the done, and then another discounted return from the 
            #reset state until the end of the trajectory (done[-1])
            if not dones[-1]:
                rewards = discount_with_dones(rewards + [value], dones[1:] + [False], GAMMA)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones[1:], GAMMA)
            mb_rewards[e_idx] = rewards

        out_mb_obs = mb_obs.reshape((-1,) + common.IMG_SHAPE)#holds observations for all envs for all trajectory steps
        out_mb_rewards = mb_rewards.flatten()#contains discounted rewards for every state of the trajectory, not just the first step
        out_mb_actions = mb_actions.flatten()
        out_mb_values = mb_values.flatten()
        out_mb_probs = mb_probs.flatten()
        yield out_mb_obs, out_mb_rewards, out_mb_actions, out_mb_values, out_mb_probs, \
              np.array(done_rewards), np.array(done_steps)


def train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values, optimizer, tb_tracker, step_idx, device="cpu"):
    optimizer.zero_grad()
    mb_adv = mb_rewards - mb_values#computes the discounted reward (for an n-step aprox trajectory) for s- value of the state
    adv_v = torch.FloatTensor(mb_adv).to(device)
    obs_v = torch.FloatTensor(mb_obs).to(device)
    rewards_v = torch.FloatTensor(mb_rewards).to(device)
    actions_t = torch.LongTensor(mb_actions).to(device)
    logits_v, values_v = net(obs_v)
    log_prob_v = F.log_softmax(logits_v, dim=1)
    log_prob_actions_v = adv_v * log_prob_v[range(len(mb_actions)), actions_t]

    loss_policy_v = -log_prob_actions_v.mean()
    loss_value_v = F.mse_loss(values_v.squeeze(-1), rewards_v)

    prob_v = F.softmax(logits_v, dim=1)
    entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()
    loss_v = ENTROPY_BETA * entropy_loss_v + VALUE_LOSS_COEF * loss_value_v + loss_policy_v
    loss_v.backward()
    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
    optimizer.step()

    tb_tracker.track("advantage", mb_adv, step_idx)
    tb_tracker.track("values", values_v, step_idx)
    tb_tracker.track("batch_rewards", rewards_v, step_idx)
    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
    tb_tracker.track("loss_value", loss_value_v, step_idx)
    tb_tracker.track("loss_total", loss_v, step_idx)
    return obs_v



def test_model(env, net, rounds=5, device="cpu"):
    total_reward = 0.0
    total_steps = 0
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)

    for _ in range(rounds):
        obs = env.reset()
        while True:
            action = agent([obs])[0][0]
            obs, r, done, _ = env.step(action)
            total_reward += r
            total_steps += 1
            if done:
                break
    return total_reward / rounds, total_steps / rounds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=False, help="Name of the run", default="_fresh")
    parser.add_argument("--seed", type=int, default=20, help="Random seed to use, default=%d" % 20)
    parser.add_argument("--steps", type=int, default=None, help="Limit of training steps, default=disabled")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
#    SEED = random.randint(0, sys.maxsize)
#    random.seed(SEED)
    #ENV_NAME = "BreakoutNoFrameskip-v4"
    
    NameForWriter = str(common.pixelsEnv) + "_" + "_" + str(common.FRAME_SIZE) + "_" + str(common.CONV_LARGE) + "_" + common.ENV_NAME + "_" + \
                    str(BATCH_SIZE) + "_" + str(REWARD_STEPS) + "_" +  str(NUM_ENVS) + "_" + str(LEARNING_RATE) + "_" + \
                    str(GAMMA) + "_" + str(ENTROPY_BETA) + "_"  + str(common.GRID_SIZE) + "_" + \
                    str(common.PARTIALLY_OBSERVED_GRID) + "_" + str(common.REPLACEMENT) + "_" + str(CLIP_GRAD) + "_" + \
                    str(SEED) + "_" + str(common.USE_FRAMESTACK_WRAPPER) + "_" + str(common.POSITIVE_RW) + "_" + str(common.NEGATIVE_RW) + "_" + str(common.SMALL_CONV_NET_CFG)
    writer = SummaryWriter(comment = "grid_01_a2c_" + NameForWriter)
    saves_path = writer.log_dir
    
    envs = [common.makeCustomizedGridEnv() for _ in range(NUM_ENVS)]

##    envs = [gym.make(ENV_NAME) for _ in range(common.NUM_ENVS)]
#    if args.seed:
#        set_seed(args.seed, envs, cuda=args.cuda)
#        suffix = "-seed=%d" % args.seed
#    else:
#        suffix = ""

    test_env = common.makeCustomizedGridEnv()

    net = common.getNet(device)
    print(common.count_parameters(net))
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    step_idx = 0
    total_steps = 0
    best_reward = None
    ts_start = time.time()
    best_test_reward = None
    with ptan.common.utils.TBMeanTracker(writer, batch_size=BATCH_SIZE) as tb_tracker:
        for mb_obs, mb_rewards, mb_actions, mb_values, _, done_rewards, done_steps in \
                iterate_batches(envs, net, device=device):
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

            train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values,
                             optimizer, tb_tracker, step_idx, device=device)
            step_idx += 1
            if args.steps is not None and args.steps < step_idx:
                break

            if step_idx % TEST_EVERY_BATCH == 0:
                test_reward, test_steps = test_model(test_env, net, device=device)
                writer.add_scalar("test_reward", test_reward, step_idx)
                writer.add_scalar("test_steps", test_steps, step_idx)
                if best_test_reward is None or best_test_reward < test_reward:
                    if best_test_reward is not None:
                        fname = os.path.join(saves_path, "best_%s_%08.3f_%d.dat" % (common.ENV_NAME,test_reward, step_idx))
                        torch.save(net.state_dict(), fname)
                    best_test_reward = test_reward
                print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                    step_idx, test_reward, test_steps, best_test_reward))
