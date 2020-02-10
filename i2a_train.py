#!/usr/bin/env python3
import os
import ptan
import time
import argparse
from tqdm import trange

import lib.trainer
import models.environment_model
from lib import common
from models import i2a_model

import torch.optim as optim
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import experiment_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CUDA", default=True, action="store_true", help="Enable CUDA")
    parser.add_argument("-e", "--EM_FILE", required=False, help="Environment model file name", default="/home/valy/OneDrive/repos/I2A-all/master/runs/Jan19_21-17-43_valy_em_22_9_False/best_1.3618e-07_184447.dat")
    parser.add_argument("--FORCE_SEED", default=False, help="Forces the use of the SEED given as an argument")
    parser.add_argument("--SEED", default=20, help="SEED for the environments")
    parser.add_argument("-g", "--GRID_SIZE", type=int, default=9, help="The size of the grid, default=5", required = False)
    parser.add_argument("-f", "--FRAME_SIZE", type=int, default=22, help="resolution of the grid, including the borders", required=False)
    parser.add_argument("-rs", "--ROLL_STEPS", type=int, default=3,help="how many steps in the imagined trajectory")
    parser.add_argument("-s","--STEPS", type=int, default=8000, help="Limit of training steps, default=disabled")
    parser.add_argument("-r","--REPLACEMENT", required=True, help="env with replacement of the squares in back in the grid")
    parser.add_argument("-lr", required = True, type=float, help="learning rate")
    parser.add_argument("-plr", required=True, type=float, help="distilled policy learning rate")

    config = experiment_config.ExperimentCfg()
    config.make_i2a_config(parser)

    device = torch.device(config.DEVICE)

    writer = SummaryWriter(comment = "_i2a_" + config.build_name_for_i2a_writer())
    saves_path = writer.logdir

    envs = [common.makeCustomizedGridEnv(config) for _ in range(config.NUM_ENVS)]
    test_env = common.makeCustomizedGridEnv(config)
    
    #sets seed on torch operations and on all environments
    common.set_seed(config.SEED, envs=envs)
    common.set_seed(config.SEED, envs=[test_env])

    obs_shape = envs[0].observation_space.shape
    act_n = envs[0].action_space.n

#    net_policy = common.AtariA2C(obs_shape, act_n).to(device)
    net_policy = common.getNet(device, config)
    config.A2CNET = str(net_policy)

    net_em = models.environment_model.EnvironmentModel(obs_shape, act_n, config)
    # net_em.load_state_dict(torch.load(config.EM_FILE_NAME, map_location=lambda storage, loc: storage))
    net_em = net_em.to(device)
    config.EM_NET = str(net_em)

    net_i2a = i2a_model.I2A(obs_shape, act_n, net_em, net_policy, config).to(device)
    config.I2A_NET = str(net_i2a)
    config.ROLLOUT_ENCODER = str(net_i2a.encoder)
#    net_i2a.load_state_dict(torch.load("saves/03_i2a_test/best_pong_-018.667_1300.dat", map_location=lambda storage, loc: storage))
#     print(net_policy)
#     print(net_em)
    print(net_i2a)
    print("em param count: ", common.count_parameters(net_em))
    print("net_policy param count: ", common.count_parameters(net_policy))
    print("ia policy param count: ", common.count_parameters(net_i2a))

    obs = envs[0].reset()
    obs_v = ptan.agent.default_states_preprocessor([obs]).to(device)
    res = net_i2a(obs_v)

    optimizer = optim.RMSprop(net_i2a.parameters(), lr=config.LEARNING_RATE, eps=1e-5)
    policy_opt = optim.Adam(net_policy.parameters(), lr=config.POLICY_LR)

    trainer = lib.trainer.A2CTrainer(envs, test_env, net_i2a, optimizer, device, config)

    epoch = 0
    total_steps = 0
    ts_start = time.time()
    best_reward = 0
    best_test_reward = 0
    # pbar = tqdm(total = config.STEPS)
    speed = 0
    mean_rw = 0
    pbar = trange(config.STEPS, desc = 'mean_rw: -, speed: -', leave=True)
    progress = iter(pbar)
    with ptan.common.utils.TBMeanTracker(writer, batch_size=32) as tb_tracker:
        for mb_obs, mb_rewards, mb_actions, mb_values, mb_probs, done_rewards, done_steps in \
                trainer.iterate_batches():
            if len(done_rewards) > 0:
                total_steps += sum(done_steps)
                speed = int(total_steps / (time.time() - ts_start))
                if best_reward == 0:
                    best_reward = done_rewards.max()
                elif best_reward < done_rewards.max():
                    best_reward = done_rewards.max()
                tb_tracker.track("total_reward_max", best_reward, epoch)
                tb_tracker.track("total_reward", done_rewards, epoch)
                tb_tracker.track("total_steps", done_steps, epoch)
                mean_rw = done_rewards.mean()
                # print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f, speed=%.2f f/s" % (
                #     epoch, len(done_rewards), mean_rw, best_reward, speed))

            obs_v = trainer.train_a2c(net_i2a, mb_obs, mb_rewards, mb_actions, mb_values,
                                     tb_tracker, epoch)
            # policy distillation
            probs_v = torch.FloatTensor(mb_probs).to(device)
            policy_opt.zero_grad()
            logits_v, _ = net_policy(obs_v)
            policy_loss_v = -F.log_softmax(logits_v, dim=1) * probs_v.view_as(logits_v)
            policy_loss_v = policy_loss_v.sum(dim=1).mean()
            policy_loss_v.backward()
            policy_opt.step()
            tb_tracker.track("loss_distill", policy_loss_v, epoch)

            epoch += 1
            desc = "speed: " + str(speed) + " mean_rw: " + str(int(mean_rw))
            pbar.set_description(desc)
            pbar.refresh()
            next(progress)
            if epoch % config.TEST_EVERY_BATCH == 0:
                test_reward, test_steps = trainer.test_model()
                writer.add_scalar("test_reward", test_reward, epoch)
                writer.add_scalar("test_steps", test_steps, epoch)
                if best_test_reward == 0 or best_test_reward < test_reward:
                    if best_test_reward > 0:
                        fname = os.path.join(saves_path, "best_%08.3f_%d.dat" % (test_reward, epoch))
                        torch.save(net_i2a.state_dict(), fname)
                        torch.save(net_policy.state_dict(), fname + ".policy")
                    else:
                        fname = os.path.join(saves_path, "em.dat")
                        torch.save(net_em.state_dict(), fname)
                    best_test_reward = test_reward
                # print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                #     epoch, test_reward, test_steps, best_test_reward))
            
            if config.STEPS is not None and config.STEPS <= epoch:
                break
# -*- coding: utf-8 -*-
        common.save_exp_config_to_TB(writer, config)