import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
import torch

from .model import EfficientZeroNet
from .env_wrapper import (
    OneHotObjEncodingWrapper,
    MinigridWrapper,
    DeterministicLavaGap,
)
from core.dataset import Transforms
from core.config import BaseConfig

games = {
    "MiniGrid-LavaGapS5-v0": {
        "return_bounds": (0.0, 0.988),
        "grid_size": 5,
        "encoded_objects": ["unseen", "wall", "empty", "goal", "lava", "agent"],
    },
    "MiniGrid-LavaGapS6-v0": {
        "return_bounds": (0.0, 0.982),
        "grid_size": 6,
        "encoded_objects": ["unseen", "wall", "empty", "goal", "lava", "agent"],
    },
    "MiniGrid-LavaGapS7-v0": {
        "return_bounds": (0.0, 0.97625),
        "grid_size": 7,
        "encoded_objects": ["wall", "empty", "goal", "lava", "agent"],
    },
    "MiniGrid-Empty-5x5-v0": {
        "return_bounds": (0.0, 0.988),
        "grid_size": 5,
        "encoded_objects": ["unseen", "wall", "empty", "goal", "agent"],
    },
}


class MinigridConfig(BaseConfig):
    def __init__(self):
        super(MinigridConfig, self).__init__(
            training_steps=20000,
            last_steps=4000,
            test_interval=500,
            log_interval=100,
            vis_interval=100,
            test_episodes=32,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=4000,
            recording_interval=100,
            max_moves=200,
            test_max_moves=220,
            history_length=400,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=50,
            batch_size=256,
            td_steps=5,
            num_actors=1,
            # network initialization/ & normalization
            init_zero=True,
            clip_reward=False,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=10000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=4,
            total_transitions=100 * 1000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=1,
            stacked_observations=4,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            # reward sum
            lstm_hidden_size=512,
            lstm_horizon_len=5,
            # siamese
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,)
        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        self.start_transitions = self.start_transitions * 1000 // self.frame_skip
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

        # Downsample observations before representation network (See paper appendix Network Architecture)
        self.downsample = self.image_based

        # Minigrid-specific parameters
        self.agent_view = False
        self.agent_view_size = 3
        self.train_proportion = .3

    def visit_softmax_temperature_fn(self, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps + self.last_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps + self.last_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name):
        self.env_name = env_name

        # Initialize environment to fetch variables
        env = gym.make(env_name)
        env.reset()
        grid = env.unwrapped.grid

        # Get the dimensions of the grid
        self.grid_height = grid.height
        self.grid_width = grid.width
        # self.grid_size = games[env_name]["grid_size"]

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
                env = RGBImgPartialObsWrapper(env)
                obs_shape = (
                    self.num_image_channels,
                    (self.agent_view_size * 2 + 1) * env.tile_size,
                    (self.agent_view_size * 2 + 1) * env.tile_size,
                )
            else:
                obs_shape = (
                    self.num_image_channels,
                    self.agent_view_size * 2 + 1,
                    self.agent_view_size * 2 + 1,
                )
        else:
            if self.image_based:
                env = RGBImgObsWrapper(env)
                obs_shape = (
                    self.num_image_channels,
                    self.grid_height * env.tile_size,
                    self.grid_width * env.tile_size,
                )
            else:
                obs_shape = (self.num_image_channels, self.grid_height, self.grid_width)
        self.obs_shape = (
            obs_shape[0] * self.stacked_observations,
            obs_shape[1],
            obs_shape[2],
        )

        # TODO: Make this calculation more general, if possible (could have a simple lookup table)
        self.num_train_levels = int(
            ((self.grid_height - 2) ** 3)
            * (self.grid_width - 4)
            * self.train_proportion
        )
        self.action_space_size = 3 if self.agent_view else 4
        self.min_return, self.max_return = games[env_name]["return_bounds"]

        # TODO: is this necessary?
        self.new_game()

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
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm)

    def new_game(self, seed=None, env_idx=0, actor_rank=0, render_mode="rgb_array",
                 record_video=False, save_path=None, recording_interval=None, test=False, final_test=False):
        if seed is None:
            seed = self.seed

        # Base environment
        if test or final_test:
            env = gym.make(
                self.env_name,
                render_mode=render_mode,
                agent_view_size=self.agent_view_size,
                max_steps=self.max_moves,
            )
        else:
            env = DeterministicLavaGap(
                seed,
                self.grid_size,
                self.max_moves,
                self.num_train_levels,
                agent_view_size=self.agent_view_size,
                render_mode=render_mode,
            )

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

        # Discard agent-centric view
        env = FullyObsWrapper(env)

        # Wrap in one-hot-encoding-wrapper
        env = OneHotObjEncodingWrapper(env, objects=self.encoded_objects)

        return MinigridWrapper(env, discount=self.discount, cvt_string=self.cvt_string)

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

game_config = MinigridConfig()
