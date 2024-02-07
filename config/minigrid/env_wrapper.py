import numpy as np
from core.game import Game
from core.utils import arr_to_str

import gymnasium as gym
from gymnasium import spaces

from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.envs import LavaGapEnv
from minigrid.wrappers import ObservationWrapper, ImgObsWrapper


class MinigridWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=True):
        """Minigrid Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.action_space.n, discount)
        self.env = ImgObsWrapper(self.env)
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        observation = observation[0].astype(np.uint8)

        done = terminated or truncated

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        (observation, _) = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()


class OneHotObjEncodingWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of the objects that are partially observable
    by the agent.
    """

    def __init__(self, env, objects=OBJECT_TO_IDX.keys()):
        """A wrapper that makes the image observation a one-hot encoding of a partially observable agent view.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.objects = objects
        if "agent" in objects:
            view_size = self.env.unwrapped.agent_view_size
            self.agent_pos = (view_size - 1, view_size // 2)

        # Number of bits per cell
        num_bits = len(objects)

        obs_shape = env.observation_space["image"].shape
        new_image_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(
            self.observation_space.spaces["image"].shape, dtype="uint8")

        col, row = self.agent_pos
        out[row, col, 0] = 1

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                if (j, i) == self.agent_pos:
                    out[i, j, self.objects.index("agent")] = 1
                else:
                    obj_idx = img[i, j, 0]
                    obj = IDX_TO_OBJECT[obj_idx]
                    out[i, j, self.objects.index(obj) + 1] = 1

        return {**obs, "image": out}
