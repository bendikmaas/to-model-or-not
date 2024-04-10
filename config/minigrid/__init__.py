import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.grid import WorldObj
from minigrid.wrappers import (
    FullyObsWrapper,
    ReseedWrapper,
)
import numpy as np
import torch

from config.minigrid.utils import WarpFrame, copy_minigrid
from .model import EfficientZeroNet
from .env_wrapper import (
    OneHotObjEncodingWrapper,
    MinigridWrapper,
    PartialOneHotObjEncodingWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    RandomizedGoalPosition,
    RandomizedStartPosition,
    WASDMinigridActionWrapper,
)
from core.dataset import Transforms
from core.config import BaseConfig

OBJECT_TO_COLOR = {
    "wall": "grey",
    "floor": "blue",
    "goal": "green",
    "lava": "red",
    "agent": "red",
    "door": "grey",
    "key": "blue",
    "ball": "blue",
    "box": "yellow",
}
N_TRAIN_LEVELS = 9

class MinigridConfig(BaseConfig):
    def __init__(self):
        super(MinigridConfig, self).__init__(
            training_steps=20 * 1000,
            last_steps=0,
            test_interval=250,
            log_interval=100,
            vis_interval=300000,
            test_episodes=32,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=20000,
            recording_interval=100,
            max_moves=200,
            test_max_moves=220,
            history_length=400,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            epsilon_max=0.99,
            epsilon_min=0.05,
            num_simulations=50,
            batch_size=256,
            td_steps=5,
            num_actors=4,
            # network initialization/ & normalization
            model_free=False,
            init_zero=True,
            clip_reward=False,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.5,
            lr_decay_steps=10000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=2000,
            total_transitions=100 * 1000,
            replay_buffer_size=50 * 1000,
            # frame skip & stack observation
            frame_skip=1,
            stacked_observations=4,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            reconstruction_coeff=1,
            # reward sum
            lstm_hidden_size=512,
            lstm_horizon_len=5,
            # siamese
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
        )
        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        # Buffer warm up
        self.start_transitions = self.start_transitions // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        if self.gray_scale:
            self.channels = 32
        self.reduced_channels_reward = 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 16  # x36 Number of channels in policy head

        # Define the hidden layers in the heads of the value functions
        self.resnet_fc_reward_layers = [32]
        self.resnet_fc_value_layers = [32]
        self.resnet_fc_policy_layers = [32]

        # Minigrid-specific parameters
        self.random_start_position = False
        self.random_goal_position = False
        self.agent_view = False
        self.agent_view_size = 5
        self.num_train_levels = N_TRAIN_LEVELS
                
        self.num_levels_per_actor = self.num_train_levels // self.num_actors

    def visit_softmax_temperature_fn(self, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps + self.last_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps + self.last_steps):
                return 0.75
            else:
                return 0.5
        else:
            return 1.0

    def epsilon_fn(self, trained_steps):
        if self.model_free:
            return max(self.epsilon_min, self.epsilon_max - (self.epsilon_decay_rate * trained_steps))
        else:
            return 0.0

    def set_game(self, env_name):
        self.env_name = env_name

        # Initialize environment to fetch variables
        env = gym.make(env_name)
        env.reset()
        grid = env.unwrapped.grid

        # Get the dimensions of the grid
        self.grid_width = grid.width
        self.grid_height = grid.height

        # Get the objects of the grid, if not image-based representation
        if self.image_based:
            self.objects_to_encode = []
            self.num_image_channels = 1 if self.gray_scale else 3
        else:
            objects = set()
            objects.add("agent")
            for col in range(grid.width):
                for row in range(grid.height):
                    obj = grid.get(col, row)
                    if obj is None:
                        objects.add("empty")
                    else:
                        objects.add(obj.type)
            if self.agent_view:
                objects.add("unseen")

            self.objects_to_encode = sorted(
                list(objects), key=lambda obj: OBJECT_TO_IDX[obj]
            )
            self.num_image_channels = len(self.objects_to_encode)

        # Determine the observation size
        if self.agent_view:
            if self.image_based:
                obs_shape = (self.num_image_channels, 96, 96)
            else:
                obs_shape = (
                    self.num_image_channels,
                    self.agent_view_size,
                    self.agent_view_size,
                )
        else:
            if self.image_based:
                obs_shape = (self.num_image_channels, 96, 96)
            else:
                obs_shape = (
                    self.num_image_channels,
                    self.grid_height,
                    self.grid_width,
                )
        self.obs_shape = (
            obs_shape[0] * self.stacked_observations,
            obs_shape[1],
            obs_shape[2],
        )

        # TODO: Make bounds more precise
        self.min_return, self.max_return = env.unwrapped.reward_range
        self.action_space_size = 3 if self.agent_view or self.image_based else 4
        
        # Calculate number of training levels so that they are scaled proportionally
        # to the number of extra levels achievable by randomizing start- and/or goal-positions
        self.num_train_levels = N_TRAIN_LEVELS
        if self.random_start_position:
            self.num_train_levels *= (self.grid_height - 2)
        if self.random_goal_position:
            self.num_train_levels *= (self.grid_height - 2)

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.do_reconstruction,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm,
        )

    def new_game(self, seed=None, env_idx=0, actor_rank=0, render_mode="rgb_array",
                 record_video=False, save_path=None, recording_interval=None, test=False, final_test=False):
        if seed is None:
            seed = self.seed

        # Base environment
        env = gym.make(
            self.env_name,
            render_mode=render_mode,
            agent_view_size=self.agent_view_size,
            max_episode_steps=self.max_moves,
        )

        if self.random_start_position:
            env = RandomizedStartPosition(env)
        if self.random_goal_position:
            env = RandomizedGoalPosition(env)

        # Remove highlight if not agent view
        if not self.agent_view:
            env.unwrapped.highlight = False

        # Wrap in video recorder
        if record_video and save_path is not None:
            from gymnasium.wrappers.record_video import RecordVideo
            if final_test:
                name_prefix = "final_test"
                interval = 1
            else:
                name_prefix = "test" if test else "train"
                interval = self.recording_interval if recording_interval is None else recording_interval
            env = RecordVideo(
                env,
                video_folder=str(save_path),
                episode_trigger=lambda episode_id: episode_id % interval == 0,
                name_prefix=name_prefix,
            )

        # Wrap according to configuration
        if self.agent_view:
            if self.image_based:
                env = RGBImgPartialObsWrapper(env)
                env = WarpFrame(
                    env,
                    height=self.obs_shape[1],
                    width=self.obs_shape[2],
                    grayscale=self.gray_scale,
                    dict_space_key="image",
                )
            else:
                env = PartialOneHotObjEncodingWrapper(
                    env, objects=self.objects_to_encode
                )
        else:
            if self.image_based:
                env = RGBImgObsWrapper(env)
                env = WarpFrame(
                    env,
                    height=self.obs_shape[1],
                    width=self.obs_shape[2],
                    grayscale=self.gray_scale,
                    dict_space_key="image",
                )
            else:
                env = WASDMinigridActionWrapper(env)
                env = FullyObsWrapper(env)
                env = OneHotObjEncodingWrapper(env, objects=self.objects_to_encode)

        # Wrap in reseed wrapper if training to only use a fixed subset of levels
        if not (test or final_test):

            # Asynchronize the seeds for each actor
            seed_idx = actor_rank * self.num_levels_per_actor
            seeds = list(range(seed, seed + self.num_train_levels))
            env = ReseedWrapper(env, seeds=seeds, seed_idx=seed_idx)

        return MinigridWrapper(
            env, self.image_based, discount=self.discount, cvt_string=self.cvt_string
        )

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(
                self.obs_shape[1], self.obs_shape[2]))

    def transform(self, images):
        return self.transforms.transform(images)

    def get_frame_from_encoded_obs(self, original_env, obs, is_reconstruction=False):
        """Get the environment frame based on a one-hot-encoded observation.

        Args:
            original_env (MinigridEnv): The original Minigrid environment.
            obs (np.ndarray): The observation of the grid. Assumes shape (W, H, C),
                                where C are one-hot encodings of objects.
        """

        # Make a copy of the environment
        env = copy_minigrid(
            original_env,
            env_name=self.env_name,
            agent_view_size=self.agent_view_size,
        )

        # Decode the observation
        W, H = env.unwrapped.grid.width, env.unwrapped.grid.height
        for i in range(W):
            for j in range(H):
                if self.agent_view:
                    # Get the relative coordinates for the subgrid
                    rel_coords = env.unwrapped.relative_coords(i, j)
                    if rel_coords is None:
                        continue
                    else:
                        col, row = rel_coords
                else:
                    col, row = i, j

                max_idx = np.argmax(obs[col, row, :-1])  # Ignore the agent
                obj = self.objects_to_encode[max_idx]

                if obj == "unseen":
                    continue
                if obj == "empty":
                    color_idx = 0
                else:
                    color_idx = COLOR_TO_IDX[OBJECT_TO_COLOR[obj]]

                obj_idx = OBJECT_TO_IDX[obj]
                world_obj = WorldObj.decode(obj_idx, color_idx, 0)
                env.unwrapped.grid.set(i, j, world_obj)

        if self.agent_view:
            env = RGBImgPartialObsWrapper(env)
        else:
            env = RGBImgObsWrapper(env)

        if not self.agent_view and is_reconstruction:

            # Sort the indices of WH in obs[-1], based on the values obs[-1]
            indices = np.argsort(obs[-1].flatten())[::-1]
            cols, rows = np.unravel_index(indices, obs[-1].shape)

            # Place the agent in the grid
            for row, col in zip(rows, cols):
                obj = env.unwrapped.grid.get(col, row)
                if obj is None or obj.can_overlap():
                    env.unwrapped.agent_pos = (cols[i], rows[i])
                    break

        frame = env.get_frame(highlight=self.agent_view)
        env.close()

        return frame


game_config = MinigridConfig()
