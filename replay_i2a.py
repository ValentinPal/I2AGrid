import ptan
import matplotlib.pyplot as plt
import torch
import numpy as np

import argparse

import gym
# import gym_RandomGoalsGrid

from models import a2c_model, i2a_model, environment_model
from experiment_config import ExperimentCfg
from lib import common
from tqdm import tqdm


def test_model(net,device, env, rounds=5):
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
    parser.add_argument("--SEED", type=int, default=20, help="Random seed to use, default=%d" % 20)
    parser.add_argument("--FORCE_SEED",type=int, default=False, help="force to use SEED")
    parser.add_argument("--CUDA", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-g", "--GRID_SIZE", type=int, default=9, help="The size of the grid, default=5", required = False)
    parser.add_argument("-f", "--FRAME_SIZE", type=int, default=22, help="resolution of the grid, including the borders", required=False)
    parser.add_argument("-r", "--REPLACEMENT", default=True, help="env with replacement of the squares in back in the grid")
    parser.add_argument("-if", "--I2A_FILE", default="runs/Jan19_23-10-12_valy_i2a_22_9_1_True/best_0015.600_3250.dat", required=False, help="")
    parser.add_argument("-a", "--A2C_FILE", default="runs/Jan19_23-10-12_valy_i2a_22_9_1_True/best_0015.600_3250.dat.policy", required=False, help="")
    parser.add_argument("-ef", "--EM_FILE", default="runs/Jan19_20-40-19_valy_em_22_9_True/best_1.4249e-06_195121.dat", required=False, help="")
    parser.add_argument("-e", "--EPISODES", default=1, type=int,required=False, help="")
    parser.add_argument("-p", "--PLOT", default=False, required=False, help="")
    parser.add_argument("-in", "--INPUT", default=False, required=False, help="")
    
    config = ExperimentCfg()
    config.make_i2a_replay_config(parser)
    device = torch.device(config.DEVICE)

    env = common.makeCustomizedGridEnv(config)
    device = torch.device("cuda")

    obs_shape = env.observation_space.shape
    act_n = env.action_space.n
    
    net = common.getNet(device, config)
    net.load_state_dict(torch.load(config.A2C_FN, map_location=lambda storage, loc: storage))
    net.to(device)

    # if(config.IS_I2A):
    net_em = environment_model.EnvironmentModel(obs_shape, act_n, config)
    net_em.load_state_dict(torch.load(config.EM_FN, map_location=lambda storage, loc: storage))
    net_em = net_em.to(device)

    net_i2a = i2a_model.I2A(obs_shape, act_n, net_em, net, config).to(device)
    net_i2a.load_state_dict(torch.load(config.I2A_FN, map_location=lambda storage, loc: storage))
    # net = net_i2a

    agent = ptan.agent.PolicyAgent(lambda x: net_i2a(x)[0], action_selector=ptan.actions.ProbabilityActionSelector(), apply_softmax=True, device=device)

    state = env.reset()
    if(config.PLOT):
        plt.imshow(np.moveaxis(state, 0 ,-1)*100, interpolation = "nearest")
        plt.show()
    
    mean_total_rw = 0
    negRws = 0
    posRws = 0
    total_steps = 0
    
    for i in tqdm(range(config.EPISODES)):
        total_rw = 0
        episode_steps = 0
        game_rw = 0
        game_count = 0
        
        while True:
            state_v = torch.tensor(np.array([state], copy=False)).to(device)
            action = agent([state])
            state, r, done, _ = env.step(action[0])
            total_rw += r
            if r < 0:
                negRws += 1
            else:
                if(r>0):
                    posRws += 1
            episode_steps += 1
            total_steps += 1
            if(config.PLOT):
                plt.imshow(np.moveaxis(state, 0 ,-1)*100, interpolation = "nearest")
                plt.show()
                print(action)
            if(config.INPUT):
                input()
                
            if done:
                state = env.reset()
                break
        
        # total_steps += episode_steps
        mean_total_rw += total_rw
    
    print("Done in %d steps, reward %.2f, negRw %.7f" % (total_steps, mean_total_rw/total_steps, negRws/total_steps))
    print(posRws, negRws, total_steps)
