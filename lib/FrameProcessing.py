from collections import deque

import gym
import numpy as np
from gym import spaces


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


def wrap_pg(env, frameSize, channels, use_stack_frames = False, stack_frames = 2, scaledFloatFrame = False):
#    env = ProcessGridFrame84(size = frameSize, env = env, channels = channels)
    if use_stack_frames:
        env = FrameStack(env, stack_frames)
#    if scaledFloatFrame:
#        env = ScaledFloatFrame(env)
    return env