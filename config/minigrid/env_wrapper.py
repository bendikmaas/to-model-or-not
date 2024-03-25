from enum import IntEnum
from typing import Any, SupportsFloat, Union
import numpy as np
from core.game import Game
from core.utils import arr_to_str


from gymnasium import Env, spaces
from gymnasium.core import Wrapper

from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.core.world_object import Goal
from minigrid.wrappers import ObservationWrapper, ImgObsWrapper


class DirectActions(IntEnum):
    # Move right, down, left or up
    right = 0
    down = 1
    left = 2
    up = 3

    # Pick up an object
    pickup = 4
    # Drop an object
    drop = 5
    # Toggle/activate an object
    toggle = 6

    # Done completing task
    done = 7


class MinigridWrapper(Game):

    def __init__(self, env, image_based, discount: float, cvt_string=True):
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

        self.image_based = image_based
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        observation = observation.astype(np.uint8)

        if self.image_based:
            # Convert from HWC to WHC
            observation = np.transpose(observation, (1, 0, 2))

        done = terminated or truncated

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        (observation, _) = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)

        if self.image_based:
            # Convert from HWC to WHC
            observation = np.transpose(observation, (1, 0, 2))

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()


class RGBImgPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        # Rendering attributes for observations
        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img_partial = self.get_frame(
            highlight=False,
            tile_size=self.tile_size,
            agent_pov=True,
        )

        return {**obs, "image": rgb_img_partial}


class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        # Rendering attributes for observations
        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)

        return {**obs, "image": rgb_img}


class PartialOneHotObjEncodingWrapper(ObservationWrapper):
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
        view_size = self.env.unwrapped.agent_view_size
        self.agent_relative_pos = (view_size - 1, view_size // 2)

        # Number of bits per cell
        num_bits = len(objects)

        obs_shape = env.observation_space["image"].shape
        new_image_space = spaces.Box(
            low=0, high=1, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, observation):
        """Take a W x H x 3 image observation and convert it to a W x H x N one-hot encoding."""
        img = observation["image"]
        out = np.zeros(
            self.observation_space.spaces["image"].shape, dtype="uint8")

        # Set the agent's position
        # TODO: This might be superfluous
        out[
            self.agent_relative_pos[0],
            self.agent_relative_pos[1],
            self.objects.index("agent"),
        ] = 1

        # Encode the rest of the grid
        for col in range(img.shape[0]):
            for row in range(img.shape[1]):
                obj_idx = img[col, row, 0]
                obj = IDX_TO_OBJECT[obj_idx]
                out[col, row, self.objects.index(obj)] = 1

        return {**observation, "image": out}


class OneHotObjEncodingWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of the full grid irrespective of the view of the agent.
    """

    def __init__(self, env, objects=list(OBJECT_TO_IDX.keys())):
        """A wrapper that makes the image observation a one-hot encoding of a complete birds-eye view of the full grid.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.objects = objects
        # Number of bits per cell / objects to encode
        num_bits = len(objects)

        obs_shape = env.observation_space["image"].shape
        new_image_space = spaces.Box(
            low=0, high=1, shape=(obs_shape[0], obs_shape[1], num_bits), dtype=np.uint8
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, observation):
        """Take a W x H x 3 image observation and convert it to a W x H x N one-hot encoding."""
        img = observation["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")

        # Determine the object at the agent's position
        col, row = self.env.unwrapped.agent_pos
        obj = self.env.unwrapped.grid.get(col, row)
        if obj is None:
            out[col, row, self.objects.index("empty")] = 1
        else:
            out[col, row, self.objects.index(obj.type)] = 1

        # Encode the rest of the grid
        for col in range(img.shape[0]):
            for row in range(img.shape[1]):
                obj_idx = img[col, row, 0]
                obj = IDX_TO_OBJECT[obj_idx]
                out[col, row, self.objects.index(obj)] = 1

        return {**observation, "image": out}


class WASDMinigridActionWrapper(Wrapper):

    def __init__(self, env: Env):
        super().__init__(env)

        # Action enumeration for Minigrid environments
        self.minigrid_actions = self.unwrapped.actions

        # Action enumeration for direct WASD control
        self.actions = DirectActions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        if action <= 3:

            while not self.unwrapped.agent_dir == action:
                if (
                    self.unwrapped.agent_dir - action == 1
                    or self.unwrapped.agent_dir - action == -3
                ):
                    super().step(self.minigrid_actions.left)
                else:
                    super().step(self.minigrid_actions.right)
                self.unwrapped.step_count -= 1
            action = self.minigrid_actions.forward
        else:
            action -= 1
        return super().step(action)


class RandomizedStartPosition(Wrapper):
    """A wrapper that randomizes the agent's starting position in the environment."""

    def __init__(
        self,
        env: Env,
        topX=1,
        topY=1,
        spawn_box_width=0,
        spawn_box_height=float("inf"),
        rand_dir=False,
    ):
        """Randomize the agent's starting position in the environment.

        Args:
            env (Env): The environment to wrap
            topX (int, optional): X-coordinate of spawn box. Defaults to 1.
            topY (int, optional): Y-coordinate of spawn box. Defaults to 1.
            spawn_box_width (int, optional): Width of spawn box. Defaults to 0.
            spawn_box_height (_type_, optional): Height of spawn boox. Defaults to float("inf").
            rand_dir (bool, optional): Randomize starting direction. Defaults to False.
        """
        super().__init__(env)

        self.top = (topX, topY)

        box_width = min(topX + spawn_box_width, self.env.unwrapped.width - topX - 1)
        box_height = min(topY + spawn_box_height, self.env.unwrapped.height - topY - 1)
        self.size = (box_width, box_height)
        self.rand_dir = rand_dir

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[dict[str, Any], None] = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self.env.unwrapped.place_agent(
            top=self.top, size=self.size, rand_dir=self.rand_dir
        )
        obs = self.env.unwrapped.gen_obs()
        return obs, {}


class RandomizedGoalPosition(Wrapper):
    """A wrapper that randomizes the goal position in the environment."""

    def __init__(self, env: Env, vertical=True):
        """Randomize the goal position in the environment.

        Args:
            env (Env): The environment to wrap
            vertical (bool, optional): Whether to randomize position vertically (or horizontally).
                                       Defaults to True.
        """
        super().__init__(env)
        self.vertical = vertical

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[dict[str, Any], None] = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        # Remove the goal from the grid
        has_goal = False
        for i in range(self.env.unwrapped.width):
            for j in range(self.env.unwrapped.height):
                if (
                    self.env.unwrapped.grid.get(i, j) is not None
                    and self.env.unwrapped.grid.get(i, j).type == "goal"
                ):
                    self.env.unwrapped.grid.set(i, j, None)
                    goalX = i
                    goalY = j
                    has_goal = True
                    break

        if has_goal:
            # The dimensions of the box in which the goal can spawn
            if self.vertical:
                spawn_box_width = 1
                spawn_box_height = self.unwrapped.height - 2
                self.env.unwrapped.goal_pos = self.env.unwrapped.place_obj(
                    Goal(), top=(goalX, 1), size=(spawn_box_width, spawn_box_height)
                )
            else:
                spawn_box_width = self.unwrapped.width - 2
                spawn_box_height = 1
                # Update goal position
                self.env.unwrapped.goal_pos = self.env.unwrapped.place_obj(
                    Goal(), top=(1, goalY), size=(spawn_box_width, spawn_box_height)
                )

        obs = self.env.unwrapped.gen_obs()
        return obs, {}
