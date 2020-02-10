#!/usr/bin/env python3
import os
import ptan
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

import models.environment_model
from lib import common
import experiment_config

def collect_experience(envs, net, config, device="cpu"):
    act_selector = ptan.actions.ProbabilityActionSelector()
    mb_obs = np.zeros((config.BATCH_SIZE, ) + config.IMG_SHAPE, dtype=np.uint8)
#    mb_obs_next = np.zeros((BATCH_SIZE, ) + i2a.EM_OUT_SHAPE, dtype=np.float32)
    mb_obs_next = np.zeros((config.BATCH_SIZE, ) + config.IMG_SHAPE, dtype=np.float32)
    mb_actions = np.zeros((config.BATCH_SIZE, ), dtype=np.int32)
    mb_rewards = np.zeros((config.BATCH_SIZE, ), dtype=np.float32)
    obs = [e.reset() for e in envs]
    total_reward = [0.0] * config.NUM_ENVS
    total_steps = [0] * config.NUM_ENVS
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
            mb_obs_next[batch_idx] = o #this is the observation after the step was taken

            mb_actions[batch_idx] = actions[e_idx]
            mb_rewards[batch_idx] = r

            total_reward[e_idx] += r
            total_steps[e_idx] += 1

            batch_idx = (batch_idx + 1) % config.BATCH_SIZE
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
    parser.add_argument("--CUDA", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--NAME", required=False, help="Name of the run", default="_fresh")
    parser.add_argument("-a", "--A2C_FILE_NAME", required=False, help="File with model to load", default="/home/valy/OneDrive/repos/I2A - all branches/master/repl/5_14/#Jan08_14-29-24_valy_a2c_14_5/best_0026.600_5250.dat")
    parser.add_argument("--FORCE_SEED", default=False, help="Forces the use of the SEED given as an argument")
    parser.add_argument("--SEED", default=20, help="SEED for the environments")
    parser.add_argument("-g", "--GRID_SIZE", type=int, default=5, help="The size of the grid, default=5", required = False)
    parser.add_argument("-f", "--FRAME_SIZE", type=int, default=14, help="resolution of the grid, including the borders", required=False)
    parser.add_argument("-s", "--EM_STEPS", type=int, required=False, default=75000)
    parser.add_argument("-r", "--REPLACEMENT", required = True, help="env with replacement of the squares in back in the grid")
    parser.add_argument("-lr", required = True, type=float,help="learning rate")
    args = parser.parse_args()

    #create configuration class that holds all configurable parameters for experimentation
    config = experiment_config.ExperimentCfg()
    config.make_em_config(parser)

    device = torch.device(config.DEVICE)
    print(config.REPLACEMENT)
    print(type(config.REPLACEMENT))
    writer = SummaryWriter(comment = "_em_" + config.build_name_for_writer())
    saves_path = writer.logdir

    #envs used for sampling tuples of experience
    envs = [common.makeCustomizedGridEnv(config) for _ in range(config.NUM_ENVS)]

    net = common.getNet(device, config)
    net.load_state_dict(torch.load(config.A2C_FILE_NAME, map_location=lambda storage, loc: storage))
    net = net.to(device)
    config.A2CNET=str(net)

#    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n)
    net_em = models.environment_model.EnvironmentModel(envs[0].observation_space.shape, envs[0].action_space.n, config).to(device)
#    net_em.load_state_dict(torch.load("/home/valy/OneDrive/experiments/repl/9_22/Jan19_20-40-19_valy_em_22_9_True/best_1.4249e-06_195121.dat", map_location=lambda storage, loc: storage))
    config.EM_NET=str(net_em)

    print(net)
    print(net_em)
    print("em param count: " + str(common.count_parameters(net_em)))

    # sets seed on torch operations and on all environments
    common.set_seed(seed=config.SEED, envs=envs)

    optimizer = optim.Adam(net_em.parameters(), lr=config.LEARNING_RATE)

    epoch = 0
    best_loss = np.inf
    desc = ""
    pbar = trange(config.EM_STEPS, desc = '', leave=True)
    progress = iter(pbar)

    with ptan.common.utils.TBMeanTracker(writer, batch_size=config.BATCH_SIZE) as tb_tracker:
        #obtain batch transitions from the a2c model free agent (st, at, st+1, r)
        for mb_obs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps in collect_experience(envs, net, config, device):
            if len(done_rewards) > 0:
                m_reward = np.mean(done_rewards)
                m_steps = np.mean(done_steps)
                # print("%d: done %d episodes, mean reward=%.2f, steps=%.2f" % (
                #     epoch, len(done_rewards), m_reward, m_steps))
                tb_tracker.track("total_reward", m_reward, epoch)
                tb_tracker.track("total_steps", m_steps, epoch)

            #transform the arrays into tensors
            #TODO: see if tensors can be used directly in the iterator, to avoid the transfer to GPU
            obs_v = torch.FloatTensor(mb_obs).to(device)
            obs_next_v = torch.FloatTensor(mb_obs_next).to(device)
            actions_t = torch.LongTensor(mb_actions.tolist()).to(device)
            rewards_v = torch.FloatTensor(mb_rewards).to(device)

            optimizer.zero_grad()

            #obtain a predictions for next state (observation) and future reward, based on current obs from the above
            #batches and the actions that are coming from the A2C model free agent
            out_obs_next_v, out_reward_v = net_em(obs_v.float(), actions_t)
            loss_obs_v = F.mse_loss(out_obs_next_v.squeeze(-1), obs_next_v)
            loss_rew_v = F.mse_loss(out_reward_v.squeeze(-1), rewards_v)

            loss_total_v = config.OBS_WEIGHT * loss_obs_v + config.REWARD_WEIGHT * loss_rew_v
            loss_total_v.backward()
            optimizer.step()
            tb_tracker.track("loss_em_obs", loss_obs_v, epoch)
            tb_tracker.track("loss_em_reward", loss_rew_v, epoch)
            tb_tracker.track("loss_em_total", loss_total_v, epoch)

            loss = loss_total_v.data.cpu().numpy()
            if loss < best_loss:
                # print("Best loss updated: %.4e -> %.4e" % (best_loss, loss))
                best_loss = loss
                fname = os.path.join(saves_path, "best_%.4e_%05d.dat" % (loss, epoch))
                torch.save(net_em.state_dict(), fname)
                loss_obs = loss_obs_v.data.cpu().numpy()
                loss_rew = loss_rew_v.data.cpu().numpy()
                desc = "loss total: %.2E " % loss + " loss obs: %.2E" % loss_obs + " loss rw: %.2E" % loss_rew

            epoch += 1
            pbar.set_description(desc)
            pbar.refresh()
            next(progress)
            if config.EM_STEPS is not None and config.EM_STEPS <= epoch:
                break

            # if epoch % config.TEST_EVERY_BATCH == 0:
            #     fname = os.path.join(saves_path, "em_%s_%05d_%.4e.dat" % (config.ENV_NAME, epoch, loss))
            #     torch.save(net_em.state_dict(), fname)

        # -*- coding: utf-8 -*-
        common.save_exp_config_to_TB(writer, config)
