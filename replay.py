from lib import gridworld_new
import ptan
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import os
import gym
import gym_RandomGoalsGrid
from time import sleep

from lib import common_new

GAMMA = 0.99
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.025
BATCH_SIZE = 50
ENV_NAME = "RandomGoalsGrid3C-v0"
GRID_SIZE = 5
PARTIALLY_OBSERVED_GRID = False
NUM_ENVS = 2
REPLACEMENT = True
REWARD_STEPS = 2
CLIP_GRAD = 1
FRAME_SIZE = 21

def makeCustomizedGridEnv():
    env = gym.make(ENV_NAME)
    env.customInit(size = GRID_SIZE,
            partial = PARTIALLY_OBSERVED_GRID,
            pixelsEnv = True,
            frameSize = FRAME_SIZE,
            replacement = REPLACEMENT,
            negativeReward = -5,
            positiveReward = 1)
    
#    env = ptan.common.wrappers.wrap_pg(env, frameSize=FRAME_SIZE, use_stack_frames= USE_FRAMESTACK_WRAPPER)
    
    return env

env = makeCustomizedGridEnv()
device = torch.device("cuda")

net = common_new.AtariA2CSmall([3,FRAME_SIZE,FRAME_SIZE], 4).to(device)
net.load_state_dict(torch.load("runs/Apr10_16-24-46_valygrid_01_a2c_True_512_21_False_RandomGoalsGrid3C-v0_64_2_32_0.0005_0.99_0.015_5_False_True_1_8511018994129316665_False_1_-1_4231/best_RandomGoalsGrid3C-v0_0026.800_11000.dat", map_location=lambda storage, loc: storage))
agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], action_selector=ptan.actions.ProbabilityActionSelector(), apply_softmax=True, device=device)

state = env.reset()
plt.imshow(np.moveaxis(env.renderEnv(), 0 ,-1), interpolation = "nearest")
plt.show()

episodes = 1000

mean_total_rw = 0
negRws = 0
for i in range(episodes):
    total_rw = 0
    steps = 0
    game_rw = 0
    game_count = 0
    
    while True:
        action, _ = agent([state])
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
        plt.imshow(np.moveaxis(env.renderEnv(), 0 ,-1), interpolation = "nearest")
        plt.show()
        input()
        if done:
            break
    
    mean_total_rw += total_rw

print("Done in %d steps, reward %.2f, negRw %.7f" % (steps, mean_total_rw/episodes, negRws/episodes))
    
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
#print(game_rw/game_count)
