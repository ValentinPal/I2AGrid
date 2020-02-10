#!/usr/bin/env python3
import os
import time
import argparse
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim

# import gym
# import gym_RandomGoalsGrid
import ptan

import lib.trainer
from lib import common
import experiment_config
from tqdm import trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CUDA", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--NAME", required=False, help="Name of the run", default="_fresh")
    parser.add_argument("--SEED", type=int, default=20, help="Random seed to use, default=%d" % 20)
    parser.add_argument("--FORCE_SEED", type=int, default=False, help="force to use SEED")
    parser.add_argument("-s", "--STEPS", type=int, default=8000, help="Limit of training steps, default=disabled")
    parser.add_argument("-g", "--GRID_SIZE", type=int, default=9, help="The size of the grid, default=5", required = False)
    parser.add_argument("-f", "--FRAME_SIZE", type=int, default=22, help="resolution of the grid, including the borders", required=False)
    parser.add_argument("-r", "--REPLACEMENT", required=True, help="env with replacement of the squares in back in the grid")
    parser.add_argument("-lr", required = True, type=float,help="learning rate")
    #create configuration class that holds all configurable parameters for experimentation
    config = experiment_config.ExperimentCfg()
    config.make_a2c_config(parser)

    device = torch.device(config.DEVICE)
    print(config.REPLACEMENT)
    print(type(config.REPLACEMENT))
    writer = SummaryWriter(comment = "_a2c_" + config.build_name_for_writer())
    saves_path = writer.logdir

    #envs used for sampling tuples of experience    
    envs = [common.makeCustomizedGridEnv(config) for _ in range(config.NUM_ENVS)]
    #env used to test the avg reward produced by the current best net
    test_env = common.makeCustomizedGridEnv(config)

    net = common.getNet(device, config)
    print(common.count_parameters(net))
    config.A2CNET = str(net)

    #sets seed on torch operations and on all environments
    common.set_seed(seed=config.SEED, envs=envs)
    common.set_seed(seed=config.SEED, envs=[test_env])
    
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

    trainer = lib.trainer.A2CTrainer(envs, test_env, net, optimizer, device, config)

    epoch = 0
    total_steps = 0
    best_reward = None
    ts_start = time.time()
    best_test_reward = 0
    best_reward = 0
    speed = 0
    mean_rw = 0
    pbar = trange(config.STEPS, desc = 'mean_rw: -, speed: -', leave=True)
    progress = iter(pbar)
    with ptan.common.utils.TBMeanTracker(writer, batch_size=config.BATCH_SIZE) as tb_tracker:
        for mb_obs, mb_rewards, mb_actions, mb_values, _, done_rewards, done_steps in \
                trainer.iterate_batches():
            if len(done_rewards) > 0:
                total_steps += sum(done_steps)
                speed = total_steps / (time.time() - ts_start)
                if best_reward is None:
                    best_reward = done_rewards.max()
                elif best_reward < done_rewards.max():
                    best_reward = done_rewards.max()
                mean_rw = done_rewards.mean()
                tb_tracker.track("total_reward_max", best_reward, epoch)
                tb_tracker.track("total_reward", done_rewards, epoch)
                tb_tracker.track("total_steps", done_steps, epoch)
                # print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f, speed=%.2f" % (
                #     epoch, len(done_rewards), done_rewards.mean(), best_reward, speed))

            trainer.train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values,
                              tb_tracker, epoch)

            desc = "speed: " + str(int(speed)) + " mean_rw: " + str(int(mean_rw))
            pbar.set_description(desc)
            pbar.refresh()
            next(progress)

            if epoch % config.TEST_EVERY_BATCH == 0:
                test_reward, test_steps = trainer.test_model()
                writer.add_scalar("test_reward", test_reward, epoch)
                writer.add_scalar("test_steps", test_steps, epoch)
                if best_test_reward is None or best_test_reward < test_reward:
                    if best_test_reward is not None:
                        fname = os.path.join(saves_path, "best_%08.3f_%d.dat" % \
                                             (test_reward, epoch))
                        torch.save(net.state_dict(), fname)
                    best_test_reward = test_reward
                # print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                #     epoch, test_reward, test_steps, best_test_reward))

            epoch += 1
            if config.STEPS is not None and config.STEPS <= epoch:
                break
        #convert the conv_layers and ImageShape dictionary to string
        common.save_exp_config_to_TB(writer, config)