import ptan
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import argparse

import gym
# import gym_RandomGoalsGrid

from models import a2cmodel, i2a_model, environment_model
from experiment_config import ExperimentCfg
from lib import common
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SEED", type=int, default=20, help="Random seed to use, default=%d" % 20)
    parser.add_argument("--FORCE_SEED", type=int, default=False, help="force to use SEED")
    parser.add_argument("--CUDA", default=True,  action="store_true", help="Enable cuda")
    parser.add_argument("-g", "--GRID_SIZE", type=int, default=5, help="The size of the grid, default=5", required=False)
    parser.add_argument("-f", "--FRAME_SIZE", type=int, default=14, help="resolution of the grid, including the borders",
                        required=False)
    parser.add_argument("-r", "--REPLACEMENT", default=True, help="env with replacement of the squares in back in the grid")
    parser.add_argument("-i", "--I2A", type=bool,default=False, required=False,
                        help="replay I2A agent, default false, which means A2C agent")
    parser.add_argument("-a", "--A2C_FILE", default="",
                        required=False, help="")
    parser.add_argument("-ef", "--EM_FILE", default="/home/valy/OneDrive/repos/I2A - all branches/master/runs/Jan18_02-20-08_valy_em_14_5_False/best_4.0376e-03_112959.dat",
                        required=False, help="")
    parser.add_argument("-e", "--EPISODES", default=1, type=int, required=False, help="")
    parser.add_argument("-p", "--PLOT", default=False, required=False, help="")
    parser.add_argument("-in", "--INPUT", default=False, required=False, help="")

    config = ExperimentCfg()
    config.make_replay_config(parser)
    device = torch.device(config.DEVICE)

    env = common.makeCustomizedGridEnv(config)
    device = torch.device("cuda")
    print(config.REPLACEMENT)
    obs_shape = env.observation_space.shape
    act_n = env.action_space.n
    #load the a2c policy used for the EM training
    net = common.getNet(device, config)
    if (config.A2C_FN != ""):
        net.load_state_dict(torch.load(config.A2C_FN, map_location=lambda storage, loc: storage))

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], action_selector=ptan.actions.ProbabilityActionSelector(),
                                   apply_softmax=True, device=device)

    net_em = environment_model.EnvironmentModel(obs_shape, act_n, config)
    net_em.load_state_dict(torch.load(config.EM_FN, map_location=lambda storage, loc: storage))
    net_em = net_em.to(device)

    state = env.reset()
    if(config.PLOT):
        plt.imshow(np.moveaxis(state, 0 ,-1)*100, interpolation = "nearest")
        plt.show()

    total_obs_loss = 0
    total_rw_loss = 0
    steps = 0
    for i in tqdm(range(config.EPISODES)):
        while True:
            action, _ = agent([state])
            state_v = torch.tensor(np.array([state], copy=True)).to(device)

            # logits_v, _ = net(state_v)
            # # probs_v = F.softmax(logits_v, dim=1)
            # probs = probs_v.data.cpu().numpy()
            # action = np.argmax(logits_v.data.cpu().numpy())
            #        #actions = act_selector(probs)

            #predict next obs and reward
            next_state_pred, next_rw_pred = net_em(state_v.float(), torch.tensor(np.array([action], copy=True)).to(device))
            #get next real obs and next real rw
            state, r, done, _ = env.step(action)

            if(config.PLOT):
                fig, ax_arr = plt.subplots(1, 2, figsize=(10, 10))
                ax_arr[0].imshow(np.moveaxis(next_state_pred[0,:,:,:].detach().cpu().numpy(), 0 ,-1), interpolation = "nearest")
                ax_arr[0].set_title("rw prediction:%.2E, action %i" %(next_rw_pred.item(),action))
                ax_arr[1].imshow(np.moveaxis(state, 0 ,-1), interpolation = "nearest")
                ax_arr[1].set_title("actual rw:%.2E" % r)
                plt.show()
            if(config.INPUT):
                input()

            total_obs_loss += F.mse_loss(next_state_pred, torch.FloatTensor(np.array([state], copy=True)).to(device))
            total_rw_loss +=  F.mse_loss(torch.FloatTensor(np.array([r], copy=True)).to(device), next_rw_pred.squeeze(-1))
            steps+=1

            if done:
                state = env.reset()
                break

    # states = torch.tensor(obs_losses)
    # rws=torch.tensor(rw_losses)
    print("obs loss:%.6E, rw loss: %.6E" % (total_obs_loss.item()/steps, total_rw_loss.item()/steps))