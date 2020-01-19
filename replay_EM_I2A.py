import ptan
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import os
import gym
import gym_RandomGoalsGrid
from models import a2c_model, i2a_model, environment_model
from experiment_config import ExperimentCfg
from lib import common
from time import sleep

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.025
BATCH_SIZE = 50
ENV_NAME = "RandomGoalsGrid3CFast-v0"
GRID_SIZE = 13
PARTIALLY_OBSERVED_GRID = False
NUM_ENVS = 2
REPLACEMENT = True
CLIP_GRAD = 1
FRAME_SIZE = 45


def makeCustomizedGridEnv():
    env = gym.make(ENV_NAME)
    env.customInit(size=GRID_SIZE,
                   partial=PARTIALLY_OBSERVED_GRID,
                   pixelsEnv=True,
                   frameSize=FRAME_SIZE,
                   replacement=REPLACEMENT,
                   negativeReward=-5,
                   positiveReward=1)
    return env

config = ExperimentCfg()
config.FRAME_SIZE = 45
device = torch.device(config.DEVICE)

env = makeCustomizedGridEnv()
device = torch.device("gpu")

obs_shape = env.observation_space.shape
act_n = env.action_space.n

net = common.getNet(device, config)
net.load_state_dict(torch.load(
    "runs/Jun06_01-30-48_valy_grid_01_a2c_RandomGoalsGrid3CFast-v0_45_13_True_64_4_32_0.001_0.99_0.015_1_3670124085/best_RandomGoalsGrid3CFast-v0_0003.600_5600.dat",
    map_location=lambda storage, loc: storage))

net_em = environment_model.EnvironmentModel(obs_shape, act_n, config)
net_em.load_state_dict(torch.load("", map_location=lambda storage, loc: storage))
net_em = net_em.to(device)

net_i2a = i2a_model.I2A(obs_shape, act_n, net_em, net, config).to(device)
net_em.load_state_dict(torch.load(
    "runs/Jun06_01-38-47_valygrid_02_a2c_RandomGoalsGrid3CFast-v0_45_13_True_64_3_32_0.0008_0.99_0.015_1_1855308529/best_RandomGoalsGrid3CFast-v0_1.6653e-07_446148.dat",
    map_location=lambda storage, loc: storage))
net_em = net_em.to(device)

net_distilled_policy = common.getNet(device, config)
net_distilled_policy.load_state_dict(torch.load(
    "runs/Jun06_08-55-50_valygrid_03_a2c_RandomGoalsGrid3CFast-v0_13_45_64_3_32_0.001_0.0001_0.99_0.015_178976524_4/best_RandomGoalsGrid3CFast-v0_0003.600_4900.dat.policy",
    map_location=lambda storage, loc: storage))


agent = ptan.agent.PolicyAgent(lambda x: net_i2a(x)[0], action_selector=ptan.actions.ProbabilityActionSelector(),
                               apply_softmax=True, device=device)

state = env.reset()
plt.imshow(np.moveaxis(env.renderEnv(), 0, -1), interpolation="nearest")
plt.show()

episodes = 100
rs = 3

actions_t = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
act_selector = ptan.actions.ProbabilityActionSelector()
# 0 - up, 1 - down, 2 - left, 3 - right
mean_total_rw = 0
mean_total_pred_rw = 0
negRws = 0
for i in range(episodes):
    total_rw = 0
    total_pred_rw = 0
    steps = 0
    game_rw = 0
    game_count = 0
    state_v = torch.tensor(dtype=torch.float32, data=np.zeros((1, 3, FRAME_SIZE, FRAME_SIZE))).to(device)
    while True:
        state_v_x = np.zeros((1, 3, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
        action, _ = agent([state])
        state_v_x[0, :, :, :] = state
        for a in range(4):
            act = [a]
            print("starting trajectory prediction for action: " + str(act[0]))
            for s in range(rs):
                print("step: " + str(s))
                pred_state, pred_rw = net_em(torch.tensor(data=state_v_x, dtype=torch.float32).to(device),
                                             torch.tensor(act, dtype=torch.int64))

                if (a == action[0]):
                    total_pred_rw += pred_rw[0]
                print("predicted next frame, action: ", act)
                plt.imshow(np.moveaxis(pred_state.detach().numpy()[0], 0, -1), interpolation="nearest")
                plt.show()
                print("predicted next reward ", pred_rw[0])
                print("sum all elements. Red: %.4f, Green:%.4f, Blue: %.4f" % (
                np.sum(pred_state[0][0].detach().numpy()), np.sum(pred_state[0][1].detach().numpy()),
                np.sum(pred_state[0][2].detach().numpy())))
                state_v_x[0, :, :, :] = np.copy(pred_state[0].detach().numpy())
                logits_v, values_v = net_distilled_policy(torch.tensor(data=state_v_x, dtype=torch.float32).to(device))
                probs_v = F.softmax(logits_v, dim=1)
                probs = probs_v.data.cpu().numpy()
                act = act_selector(probs)

            input()
            state_v_x[0, :, :, :] = state
        #        state_v = torch.tensor(np.array([state], copy=False)).to(device)
        #        logits_v, _ = net(state_v)
        #        probs_v = F.softmax(logits_v, dim=1)
        #        probs = probs_v.data.cpu().numpy()
        #        action = np.argmax(probs)
        #        #actions = act_selector(probs)

        state, r, done, _ = env.step(action)
        total_rw += r
        if r < 0:
            negRws += 1
        steps += 1
        print("Actual next frame")
        plt.imshow(np.moveaxis(env.renderEnv(), 0, -1), interpolation="nearest")
        plt.show()
        print("chosen action is:" + str(action[0]))
        input()
        if done:
            break

    mean_total_rw += total_rw
    mean_total_pred_rw += total_pred_rw

print("Done in %d steps, reward %.2f, negRw %.7f, pred_rw %.3f" % (
steps, mean_total_rw / episodes, negRws / episodes, mean_total_pred_rw / episodes))

#
#    action, _ = agent([state])
#    state, reward, done = env.step(action)
#    total_rw += reward
#    steps += 1
#    renderedEnv = env.renderEnv()
#    plt.imshow(np.moveaxis(env.renderEnv(), 0 ,-1), interpolation = "nearest")
#    plt.show()
#
#    if done:
#        game_rw += total_rw
#        game_count += 1
#        print (total_rw, steps)
#        steps = 0
#        total_rw = 0
#        state = env.reset()
#
#    input()
##    if(a==5):
##        break
#
# print(game_rw/game_count)
