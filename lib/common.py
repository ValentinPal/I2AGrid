from collections import deque
import gym
import ptan
import numpy as np
import time
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from numba import jit
from gym import spaces

DEFAULT_SEED = 20

NUM_ENVS = 32
GAMMA = 0.99
REWARD_STEPS = 1
ENTROPY_BETA = 0.015
VALUE_LOSS_COEF = 0.5
BATCH_SIZE = 32
CLIP_GRAD = 1

channels = 3
FRAMES_COUNT = 1
FRAME_SIZE = 33
IMG_SHAPE = (channels * FRAMES_COUNT, FRAME_SIZE, FRAME_SIZE)

ENV_NAME = "RandomGoalsGrid3CFast-v0"
GRID_SIZE = 9
PARTIALLY_OBSERVED_GRID = False
USE_FRAMESTACK_WRAPPER = True
CONV_LARGE = True
pixelsEnv = True
REPLACEMENT = True
NEGATIVE_RW = -1
POSITIVE_RW = 1
SMALL_CONV_NET_CFG = "524231_16_32"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AtariA2CLarge(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2CLarge, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)
    
class AtariA2CSmall(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2CSmall, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float()# / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)
    
class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

def getNet(device):
    net = None
    if pixelsEnv:
        if CONV_LARGE:
            net = AtariA2CLarge(IMG_SHAPE, 4).to(device)
        else:
            net = AtariA2CSmall(IMG_SHAPE, 4).to(device)
    else:
        net = MatrixA2C(GRID_SIZE ** 2, 4).to(device)
    print(net)
    return net

def makeCustomizedGridEnv():
    env = gym.make(ENV_NAME)
    env.customInit(size = GRID_SIZE,
            partial = PARTIALLY_OBSERVED_GRID,
            pixelsEnv = pixelsEnv,
            frameSize = FRAME_SIZE,
            replacement = REPLACEMENT,
            negativeReward = NEGATIVE_RW,
            positiveReward = POSITIVE_RW)
    
#    env = wrap_pg(env, channels = channels, frameSize=FRAME_SIZE, use_stack_frames= USE_FRAMESTACK_WRAPPER)
    
    return env

def make_env(test=False, clip=True, env_name = "BreakoutNoFrameskip-v4"):
    if test:
        args = {'reward_clipping': False,
                'episodic_life': False}
    else:
        args = {'reward_clipping': clip}
    return ptan.common.wrappers.wrap_dqn(gym.make(env_name),
                                         stack_frames=FRAMES_COUNT,
                                         **args)



def set_seed(seed, envs=None, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    if envs:
        for idx, env in enumerate(envs):
            env.seed(seed + idx)

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def iterate_batches(envs, net, device="gpu"):
    n_actions = envs[0].action_space.n
    act_selector = ptan.actions.ProbabilityActionSelector()
    obs = [e.reset() for e in envs]
    batch_dones = [[False] for _ in range(NUM_ENVS)]
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    mb_obs = np.zeros((NUM_ENVS, REWARD_STEPS) + IMG_SHAPE, dtype=np.uint8)
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

        out_mb_obs = mb_obs.reshape((-1,) + IMG_SHAPE)#holds observations for all envs for all trajectory steps
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


def wrap_pg(env, frameSize, channels, use_stack_frames = False, stack_frames = 2, scaledFloatFrame = False):
#    env = ProcessGridFrame84(size = frameSize, env = env, channels = channels)
    if use_stack_frames:
        env = FrameStack(env, stack_frames)
#    if scaledFloatFrame:
#        env = ScaledFloatFrame(env)
    return env

class ProcessGridFrame84(gym.ObservationWrapper):
    def __init__(self, size, channels, env=None):
        super(ProcessGridFrame84, self).__init__(env)
        self.size = size
        self.channels = channels
        self.observation_space = spaces.Box(low=0, high=1, shape=(channels, self.size, self.size), dtype=np.float32)

    def observation(self, obs):
        return ProcessGridFrame84.process(obs, self.size, self.channels)

    @staticmethod
    def process(frame, size, channels):
        img = frame.astype(np.float32)
#        if frame.size == 210 * 160 * 3:
#            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
#        elif frame.size == 250 * 160 * 3:
#            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
#        else:
#            assert False, "Unknown resolution."
        img = img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2, :, :] * 0.114
        img = np.reshape(img, [1, size, size])
#        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
#        x_t = resized_screen[18:102, :]
#        x_t = np.reshape(x_t, [84, 84, 1])
#        return x_t.astype(np.uint8)
        return img

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=1, shape=(shp[0]*k, shp[1], shp[2]), dtype=np.float32)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
