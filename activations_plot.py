import argparse

import ptan
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import gym
import gym_RandomGoalsGrid
from torch.nn import functional as F

from experiment_config import ExperimentCfg
from lib import common
from models import environment_model, i2a_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SEED", type=int, default=20, help="Random seed to use, default=%d" % 20)
    parser.add_argument("--FORCE_SEED", type=int, default=False, help="force to use SEED")
    parser.add_argument("--CUDA", default=False,  action="store_true", help="Enable cuda")
    parser.add_argument("-g", "--GRID_SIZE", type=int, default=5, help="The size of the grid, default=5", required=False)
    parser.add_argument("-f", "--FRAME_SIZE", type=int, default=14, help="resolution of the grid, including the borders",
                        required=False)
    parser.add_argument("-r", "--REPLACEMENT", default=True, help="env with replacement of the squares in back in the grid")
    parser.add_argument("-mode", "--MODE", default="i2a", help="i2a or a2c are possible values")
    parser.add_argument("-if", "--I2A_FILE", default="runs/Jan19_23-10-12_valy_i2a_22_9_1_True/best_0015.600_3250.dat", required=False, help="")
    parser.add_argument("-a", "--A2C_FILE", default="runs/Jan19_23-10-12_valy_i2a_22_9_1_True/best_0015.600_3250.dat.policy", required=False, help="")
    parser.add_argument("-ef", "--EM_FILE", default="runs/Jan19_20-40-19_valy_em_22_9_True/best_1.4249e-06_195121.dat", required=False, help="")

    config = ExperimentCfg()
    config.make_grad_cam_config(parser)
    device = torch.device("cpu")

    env = common.makeCustomizedGridEnv(config)

    obs_shape = env.observation_space.shape
    act_n = env.action_space.n

    net = common.getNet(device, config)
    net.load_state_dict(torch.load(config.A2C_FN, map_location=lambda storage, loc: storage))
    net.to(device)
    net.eval()
    net.grad_cam = True
    agent = None

    if config.MODE == "i2a":
        net_em = environment_model.EnvironmentModel(obs_shape, act_n, config)
        net_em.load_state_dict(torch.load(config.EM_FN, map_location=lambda storage, loc: storage))
        net_em = net_em.to(device)
        net_em.eval()

        net_i2a = i2a_model.I2A(obs_shape, act_n, net_em, net, config).to(device)
        net_i2a.load_state_dict(torch.load(config.I2A_FN, map_location=lambda storage, loc: storage))
        net_i2a.eval()
        net_i2a.grad_cam = True
        net.grad_cam = True

        agent = ptan.agent.PolicyAgent(lambda x: net_i2a(x)[0], action_selector=ptan.actions.ProbabilityActionSelector(), apply_softmax=True, device=device)
    else:
        agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], action_selector=ptan.actions.ProbabilityActionSelector(), apply_softmax=True, device=device)

    state = env.reset()
    while 1 == 1:
        for act in range(4):
            state_v = torch.tensor(np.array([state], copy=False)).to(device)

            # model_activations = SaveActivations(net.conv)
            logits_action_v, _ = net(state_v)
            probs_v = F.softmax(logits_action_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            action = np.argmax(probs)
            logits_action_v[0][act].backward()

            grad = net.get_activations_gradient()
            pooled_gradients = torch.mean(grad, dim=[0, 2, 3])

            activations = net.get_activations(state_v).detach()

            for i in range(64):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= torch.max(heatmap)

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
            print(act, probs.squeeze()[act])
            print(heatmap.squeeze())
            stateImg = np.moveaxis(env.renderEnv(), 0, -1)
            ax0.imshow(stateImg, interpolation="nearest")
            ax1.imshow(heatmap.squeeze())

            heatmap = cv2.resize(heatmap.cpu().numpy(), (config.FRAME_SIZE, config.FRAME_SIZE), interpolation=cv2.INTER_NEAREST)

            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.2 + stateImg
            cv2.imwrite(f"./map{act}.jpg", superimposed_img)
            img = cv2.imread(f"./map{act}.jpg")
            plt.imshow(img)

            plt.show()

        state, done, _, _ = env.step(action)
        a = input()
        if a == 7: break