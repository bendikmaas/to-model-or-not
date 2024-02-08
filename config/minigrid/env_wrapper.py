import numpy as np
from core.game import Game
from core.utils import arr_to_str

import random

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
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


class DeterministicLavaGap(LavaGapEnv):
    def __init__(
        self, seed, size, max_steps, num_train_levels, **kwargs
    ):
        super().__init__(size, max_steps=max_steps, **kwargs)

        # Enumerate and store all possible gap positions
        self.legal_gap_positions = []
        for row in range(1, size - 1):
            for col in range(2, size - 2):
                self.legal_gap_positions.append((col, row))

        self.num_train_levels = min(
            num_train_levels, len(self.legal_gap_positions))

        rng = random.Random(seed)
        self.gap_positions = rng.sample(self.legal_gap_positions,
                                        self.num_train_levels)
        print(f"Selected gap positions: {self.gap_positions}")

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # Generate and store random gap positionn
        self.gap_pos = np.array(self._rand_elem(self.gap_positions))

        # Place the obstacle wall
        self.grid.vert_wall(self.gap_pos[0], 1, height - 2, self.obstacle_type)

        # Put a hole in the wall
        self.grid.set(*self.gap_pos, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
