import argparse

import matplotlib.pyplot as plt
import numpy as np
import ptan
import torch
from tqdm import tqdm

from experiment_config import ExperimentCfg
from lib import common

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--GRID_SIZE", type=int, default=5, help="The size of the grid, default=5", required = False)
    parser.add_argument("--SEED", type=int, default=20, help="Random seed to use, default=%d" % 20)
    parser.add_argument("--CUDA", default=True,  action="store_true", help="Enable cuda")
    parser.add_argument("--FORCE_SEED", type=int, default=False, help="force to use SEED")
    parser.add_argument("-f", "--FRAME_SIZE", type=int, default=14, help="resolution of the grid, including the borders", required=False)
    parser.add_argument("-r", "--REPLACEMENT", default=False, help="env with replacement of the squares in back in the grid")
    parser.add_argument("-in", "--INPUT", type=bool, default=True, required=False, help="")
    parser.add_argument("-e", "--EPISODES", default=3, type=int,required=False, help="")
    parser.add_argument("-a", "--A2C_FILE", default="/home/valy/OneDrive/repos/I2A - all branches/master/runs/Jan18_15-45-16_valy_a2c_14_5_False/best_0004.000_3500.dat",
                        required=False, help="")
    parser.add_argument("-p", "--PLOT", default=False,required=False, help="")
    parser.add_argument("-lr", required=True, type=float, help="learning rate")

    fig, _ = plt.subplots()

    config = ExperimentCfg()
    config.make_test_env_config(parser)
    device = torch.device(config.DEVICE)

    env = common.makeCustomizedGridEnv(config)
    device = torch.device("cuda")

    obs_shape = env.observation_space.shape
    act_n = env.action_space.n

    net = common.getNet(device, config)
    net.load_state_dict(torch.load(config.A2C_FN, map_location=lambda storage, loc: storage))

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], action_selector=ptan.actions.ProbabilityActionSelector(),
                                   apply_softmax=True, device=device)

    state = env.reset()

    total_rw = 0
    total_steps = 0
    episodes = config.EPISODES
    for i in tqdm(range(episodes)):
        while True:
            action, _ = agent([state])
            if(config.PLOT):
                plt.imshow(np.moveaxis(state, 0 ,-1), interpolation = "nearest")
                plt.show()
                # action = input()
            state, r, done, _ = env.step(action)
            total_rw += r
            total_steps += 1
            if done:
                state = env.reset()
                break

    print(total_rw, total_steps)