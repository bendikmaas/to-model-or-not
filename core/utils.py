import os
import cv2
import gym
import torch
import random
import shutil
import logging

import numpy as np

from scipy.stats import entropy


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Induce some variation into start positions taking random number of no-ops on reset.
        
        Note:
            No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame and 
        max-pool over the last two frames to percieve Atari's flickering assets."""
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action '_skip' times, sum up reward, and compute the 
            max pixel values over the two last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        img = self.max_frame
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


def make_atari(env, skip=4, max_episode_steps=None):
    """Make Atari games
    Parameters
    ----------
    env: AtariEnv
        base Atari environment
    skip: int
        frame skip
    max_episode_steps: int
        max moves for an episode
    """
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_results_dir(exp_path, args):
    # make the result directory
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force and not args.auto_resume:
            raise FileExistsError('{} is not empty. Please use --force to overwrite it'.format(exp_path))
        elif args.force and not args.auto_resume:
            print('Warning, path exists! Rewriting...')
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
    return exp_path, log_path


def init_logger(base_path):
    # initialize the logger
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def softmax(logits):
    """Compute softmax probabilities from logits."""
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits)

def select_action(logits, model_free, temperature=1, deterministic=True, epsilon=0.0):
    """select action from the root visit counts.
    Parameters
    ----------
    model_free: bool
        True -> model free
    temperature: float
        the temperature for the distribution
    deterministic: bool
        True -> select the argmax
        False -> sample from the distribution
    """
    if model_free:
        action_probs = softmax(np.array(logits) / temperature)
    else:
        temperatured_logits = [
            logit ** (1 / temperature) for logit in logits
        ]
        total_sum = np.sum(temperatured_logits)
        action_probs = [x / total_sum for x in temperatured_logits]
    
    if deterministic:
        action_pos = np.argmax(action_probs)
    #elif model_free and np.random.rand() < epsilon:
    #    action_pos = np.random.choice(len(logits))
    else:
        action_pos = np.random.choice(len(logits), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return action_pos, count_entropy


def prepare_observation_lst(observation_lst):
    """Prepare the observations to satisfy the input fomat of torch
    [B, S, W, H, C] -> [B, S x C, W, H]
    batch, stack num, width, height, channel
    """
    # B, S, W, H, C
    observation_lst = np.array(observation_lst, dtype=np.uint8)
    observation_lst = np.moveaxis(observation_lst, -1, 2)

    shape = observation_lst.shape
    observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


def arr_to_str(arr):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function encodes the observation numpy arr to the jpeg strings
    """
    img_str = cv2.imencode('.jpg', arr)[1].tobytes()

    return img_str


def str_to_arr(s, gray_scale=False):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function decodes the observation numpy arr from the jpeg strings
    Parameters
    ----------
    s: string
        the inputs
    gray_scale: bool
        True -> the inputs observation is gray not RGB.
    """
    nparr = np.frombuffer(s, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return arr
