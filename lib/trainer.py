import numpy as np
import ptan
import torch
from torch.nn import functional as F, utils as nn_utils

class A2CTrainer:
    def __init__(self, envs, test_env, net, optimizer, device, cfg):
        self.envs = envs
        self.test_env = test_env
        self.net = net
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

    def iterate_batches(self):
        n_actions = self.envs[0].action_space.n
        act_selector = ptan.actions.ProbabilityActionSelector()
        obs = [e.reset() for e in self.envs]
        batch_dones = [[False] for _ in range(self.cfg.NUM_ENVS)]
        total_reward = [0.0] * self.cfg.NUM_ENVS
        total_steps = [0] * self.cfg.NUM_ENVS
        mb_obs = np.zeros((self.cfg.NUM_ENVS, self.cfg.REWARD_STEPS) + self.cfg.IMG_SHAPE, dtype=np.uint8)
        mb_rewards = np.zeros((self.cfg.NUM_ENVS, self.cfg.REWARD_STEPS), dtype=np.float32)
        mb_values = np.zeros((self.cfg.NUM_ENVS, self.cfg.REWARD_STEPS), dtype=np.float32)
        mb_actions = np.zeros((self.cfg.NUM_ENVS, self.cfg.REWARD_STEPS), dtype=np.int32)
        mb_probs = np.zeros((self.cfg.NUM_ENVS, self.cfg.REWARD_STEPS, n_actions), dtype=np.float32)

        # one iteration computes partial trajectories (n step) for all environments
        # 1. forward pass to get action logits and V(s) for a specified observation
        # 2. select the action based on action selector (using action probabilities given by the model) and executes it
        # 3. store rewards per step per env, total reward per env (per trajectory), "dones", last obs
        # 4. compute total discounted reward for the n-step trajectory using V(s)*(gamma**n) at the end if trajectory did not reach "done"
        # 5. yield returns last observations per trajectory(per env), actions taken, V(s), a list of total rewards for done episodes,
        # a list of the steps where episodes finished
        # next time the iterator calls this method, this function will do another while iteration
        while True:
            batch_dones = [[dones[-1]] for dones in batch_dones]
            done_rewards = []
            done_steps = []
            for n in range(self.cfg.REWARD_STEPS):  # interact with all envs for every step of a partial trajectory
                obs_v = ptan.agent.default_states_preprocessor(obs).to(self.device)
                mb_obs[:, n] = obs_v.data.cpu().numpy()
                logits_v, values_v = self.net(
                    obs_v)  # forward pass, to get the logits for the actions and the V of the state/obs
                probs_v = F.softmax(logits_v,
                                    dim=1)  # probabilities of the logits from the AC (it can also be i2a) model
                probs = probs_v.data.cpu().numpy()
                actions = act_selector(
                    probs)  # selects an action by sampling according to the probs given by AC model(or i2a)
                mb_probs[:, n] = probs
                mb_actions[:, n] = actions  # one action per environment, for the current step of the trajectory
                mb_values[:, n] = values_v.squeeze().data.cpu().numpy()  # value for every state from the trajectory
                for e_idx, e in enumerate(self.envs):
                    o, r, done, _ = e.step(actions[e_idx])
                    total_reward[e_idx] += r  # computes partial trajectory rw per each env simulation
                    total_steps[e_idx] += 1  # trajectory steps per env
                    if done:
                        o = e.reset()
                        done_rewards.append(total_reward[e_idx])
                        done_steps.append(total_steps[e_idx])
                        total_reward[e_idx] = 0.0
                        total_steps[e_idx] = 0
                    obs[e_idx] = o
                    mb_rewards[e_idx, n] = r  # reward per env and per step for a trajectory
                    batch_dones[e_idx].append(done)
            # obtain values for the last observation. obs holds last observation for every environment
            obs_v = ptan.agent.default_states_preprocessor(obs).to(self.device)
            _, values_v = self.net(obs_v)
            values_last = values_v.squeeze().data.cpu().numpy()

            # compute the total discounted reward for an n-step trajectory, per environment
            # if last step is not end of trajectory, add to the rewards V(last_obs_of_traj)
            for e_idx, (rewards, dones, value) in enumerate(zip(mb_rewards, batch_dones, values_last)):
                rewards = rewards.tolist()  # all rewards gotten for every step of this trajectory for this env
                # TODO: what if done is at the middle of a trajectory and then another one starts for the same env?
                # shouldn t we compute discounted reward up to the done, and then another discounted return from the
                # reset state until the end of the trajectory (done[-1])
                if not dones[-1]:
                    rewards = self.discount_with_dones(rewards + [value], dones[1:] + [False], self.cfg.GAMMA)[:-1]
                else:
                    rewards = self.discount_with_dones(rewards, dones[1:], self.cfg.GAMMA)
                mb_rewards[e_idx] = rewards

            out_mb_obs = mb_obs.reshape((-1,) + self.cfg.IMG_SHAPE)  # holds observations for all envs for all trajectory steps
            out_mb_rewards = mb_rewards.flatten()  # contains discounted rewards for every state of the trajectory, not just the first step
            out_mb_actions = mb_actions.flatten()
            out_mb_values = mb_values.flatten()
            out_mb_probs = mb_probs.flatten()

            yield out_mb_obs, out_mb_rewards, out_mb_actions, out_mb_values, out_mb_probs, \
                  np.array(done_rewards), np.array(done_steps)


    def train_a2c(self, net, mb_obs, mb_rewards, mb_actions, mb_values, tb_tracker, step_idx):
        self.optimizer.zero_grad()
        mb_adv = mb_rewards - mb_values  # computes the discounted reward (for an n-step aprox trajectory) for s- value of the state
        adv_v = torch.FloatTensor(mb_adv).to(self.device)
        obs_v = torch.FloatTensor(mb_obs).to(self.device)
        rewards_v = torch.FloatTensor(mb_rewards).to(self.device)
        actions_t = torch.LongTensor(mb_actions).to(self.device)
        logits_v, values_v = net(obs_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = adv_v * log_prob_v[range(len(mb_actions)), actions_t]

        loss_policy_v = -log_prob_actions_v.mean()
        loss_value_v = F.mse_loss(values_v.squeeze(-1), rewards_v)

        prob_v = F.softmax(logits_v, dim=1)
        entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()
        loss_v = self.cfg.ENTROPY_BETA * entropy_loss_v + self.cfg.VALUE_LOSS_COEF * loss_value_v + loss_policy_v
        loss_v.backward()
        nn_utils.clip_grad_norm_(net.parameters(), self.cfg.CLIP_GRAD)
        self.optimizer.step()

        tb_tracker.track("advantage", mb_adv, step_idx)
        tb_tracker.track("values", values_v, step_idx)
        tb_tracker.track("batch_rewards", rewards_v, step_idx)
        tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
        tb_tracker.track("loss_policy", loss_policy_v, step_idx)
        tb_tracker.track("loss_value", loss_value_v, step_idx)
        tb_tracker.track("loss_total", loss_v, step_idx)
        return obs_v


    def test_model(self, rounds=5):
        total_reward = 0.0
        total_steps = 0
        agent = ptan.agent.PolicyAgent(lambda x: self.net(x)[0], device=self.device, apply_softmax=True)

        for _ in range(rounds):
            obs = self.test_env.reset()
            while True:
                action = agent([obs])[0][0]
                obs, r, done, _ = self.test_env.step(action)
                total_reward += r
                total_steps += 1
                if done:
                    break
        return total_reward / rounds, total_steps / rounds


    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]
